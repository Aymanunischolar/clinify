"""LLM provider abstraction shared by all agent nodes.

Runs against Google Gemini when GEMINI_API_KEY is set (tries GEMINI_MODEL
first, then falls through GEMINI_FALLBACK_MODELS on failure/quota errors).
Otherwise falls back to a deterministic mock provider so the full graph
(planning, retrieval, reasoning, writing, QA) can be exercised end-to-end
without any API key — useful for demos, tests, and CI.
"""
from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", "").split(",") if m.strip()
]


class LLMError(RuntimeError):
    pass


def _mock_structured_response(system_prompt: str, user_prompt: str, schema_hint: str) -> dict[str, Any]:
    """Best-effort deterministic mock for offline/demo mode.

    Inspects which agent role is calling (via the system prompt) and
    returns a plausible structured payload so downstream nodes still get
    well-formed JSON to work with.
    """
    role = system_prompt.lower()

    if "planner" in role:
        return {
            "sub_questions": [user_prompt.strip()[:200]],
            "search_queries": [user_prompt.strip()[:200]],
            "requires_coding": any(
                kw in user_prompt.lower() for kw in ["diagnos", "code", "icd", "bill"]
            ),
        }

    if "reason" in role:
        return {
            "key_findings": [
                "Mock mode: no GEMINI_API_KEY configured, so this is a "
                "template reasoning summary generated from retrieved context "
                "rather than an LLM. Set GEMINI_API_KEY for real reasoning."
            ],
            "supporting_chunk_ids": [],
        }

    if "writer" in role:
        return {
            "answer": (
                "[MOCK MODE — no GEMINI_API_KEY set] Based on the retrieved "
                "clinical guideline excerpts below, here is a template "
                "response. Configure GEMINI_API_KEY to get a real generated "
                "answer grounded in the retrieved context."
            ),
            "citations": [],
        }

    if "qa" in role or "verif" in role:
        return {
            "is_grounded": True,
            "faithfulness_notes": "Mock mode: verification skipped, assumed grounded.",
            "revised_answer": None,
        }

    if "coding" in role or "icd" in role:
        return {"suggested_codes": [], "rationale": "Mock mode: no LLM configured."}

    return {"raw": "Mock response — no GEMINI_API_KEY configured."}


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
        self.mock = not self.api_key
        self._genai_client = None

    def _client(self):
        if self._genai_client is None:
            from google import genai

            self._genai_client = genai.Client(api_key=self.api_key)
        return self._genai_client

    def structured_completion(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_hint: str = "Return valid JSON.",
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        """Returns a parsed JSON dict from the LLM (or the mock provider)."""
        if self.mock:
            return _mock_structured_response(system_prompt, user_prompt, schema_hint)

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
        raise LLMError(f"All Gemini models failed ({self.models_to_try}): {last_error}") from last_error

    def stream_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.3):
        """Yields text chunks. In mock mode, yields the mock answer in pieces."""
        if self.mock:
            text = _mock_structured_response(system_prompt, user_prompt, "")["answer"] \
                if "writer" in system_prompt.lower() else \
                "[MOCK MODE — no GEMINI_API_KEY set] " + user_prompt[:200]
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
        raise LLMError(f"All Gemini models failed ({self.models_to_try}): {last_error}") from last_error


_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
