"""LLM provider abstraction shared by all agent nodes.

Runs against Google Gemini when GEMINI_API_KEY is set (tries GEMINI_MODEL
first, then falls through GEMINI_FALLBACK_MODELS on failure/quota errors).
Otherwise falls back to a deterministic mock provider so the full graph
(planning, retrieval, reasoning, writing, QA) can be exercised end-to-end
without any API key — useful for demos, tests, and CI.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("clinicalrag.llm")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
]

_MOCK_RESPONSES = {
    "planner": lambda user_prompt: {
        "sub_questions": [user_prompt.strip()[:200]],
        "search_queries": [user_prompt.strip()[:200]],
        "requires_coding": any(
            kw in user_prompt.lower() for kw in ["diagnos", "code", "icd", "bill"]
        ),
    },
    "reasoner": lambda user_prompt: {
        "key_findings": [
            "Mock mode: no GEMINI_API_KEY configured, so this is a "
            "template reasoning summary generated from retrieved context "
            "rather than an LLM. Set GEMINI_API_KEY for real reasoning."
        ],
        "supporting_chunk_ids": [],
    },
    "writer": lambda user_prompt: {
        "answer": (
            "[MOCK MODE — no GEMINI_API_KEY set] Based on the retrieved "
            "clinical guideline excerpts below, here is a template "
            "response. Configure GEMINI_API_KEY to get a real generated "
            "answer grounded in the retrieved context."
        ),
        "citations": [],
    },
    "qa": lambda user_prompt: {
        "is_grounded": True,
        "faithfulness_notes": "Mock mode: verification skipped, assumed grounded.",
        "revised_answer": None,
    },
    "coding": lambda user_prompt: {"suggested_codes": [], "rationale": "Mock mode: no LLM configured."},
}


def _mock_structured_response(role: str, user_prompt: str) -> dict[str, Any]:
    """Deterministic mock for offline/demo mode, keyed by an explicit agent
    role (not sniffed from prompt text, which is fragile — e.g. the Writer
    prompt mentions "reasoning summary" and would false-match a "reasoner"
    substring check)."""
    builder = _MOCK_RESPONSES.get(role)
    if builder is None:
        return {"raw": "Mock response — no GEMINI_API_KEY configured."}
    return builder(user_prompt)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.mock = not self.api_key or os.getenv("FORCE_MOCK_LLM") == "1"
        self._genai_client = None

    def _client(self):
        if self._genai_client is None:
            from google import genai

            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    def structured_completion(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        schema_hint: str = "Return valid JSON.",
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Returns a parsed JSON dict from the LLM (or the mock provider).

        `role` identifies the calling agent (e.g. "planner", "writer") and
        is used only for the offline mock provider's response shape.
        """
        if self.mock:
            return _mock_structured_response(role, user_prompt)

        from google.genai import types

        client = self._client()
        last_error: Exception | None = None
        for model_name in self.models_to_try:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=f"{system_prompt}\n\n{schema_hint}",
                        temperature=temperature,
                        response_mime_type="application/json",
                    ),
                )
                return _extract_json(response.text)
            except Exception as e:  # noqa: BLE001 - try next model in the fallback chain
                last_error = e
                continue

        # All configured Gemini models failed (e.g. provider outage or quota
        # exhaustion). Degrade to the mock provider rather than 500ing the
        # whole request — a rate-limited LLM shouldn't take down retrieval,
        # citations, or the rest of the pipeline.
        logger.warning(
            "All Gemini models failed (%s): %s — degrading to mock response",
            self.models_to_try,
            last_error,
        )
        return _mock_structured_response(role, user_prompt)

    def stream_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.3):
        """Yields text chunks of a Writer-style prose answer. In mock mode,
        yields the mock Writer answer in pieces."""
        if self.mock:
            text = _mock_structured_response("writer", user_prompt)["answer"]
            for i in range(0, len(text), 24):
                yield text[i : i + 24]
            return

        from google.genai import types

        client = self._client()
        last_error: Exception | None = None
        for model_name in self.models_to_try:
            try:
                stream = client.models.generate_content_stream(
                    model=model_name,
                    contents=user_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=temperature,
                    ),
                )
                for chunk in stream:
                    if chunk.text:
                        yield chunk.text
                return
            except Exception as e:  # noqa: BLE001 - try next model in the fallback chain
                last_error = e
                continue

        logger.warning(
            "All Gemini models failed (%s): %s — degrading to mock stream",
            self.models_to_try,
            last_error,
        )
        text = "[MOCK MODE — Gemini unavailable] " + user_prompt[:200]
        for i in range(0, len(text), 24):
            yield text[i : i + 24]


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
