"""CrewAI crew: a role-based secondary workflow for ICD-10 coding
suggestions, distinct from the primary LangGraph pipeline. It shares the
same hybrid retrieval layer as the main graph, and demonstrates a second
agentic framework plus inter-agent tool use (ICD-10 lookup tool).

Two roles, run via the actual CrewAI framework (Agent/Task/Crew) against
Gemini through litellm's "gemini/<model>" provider prefix:
  - Coding Agent: proposes candidate ICD-10 codes from retrieved context
    and the ICD-10 lookup tool.
  - Verification Agent: cross-checks the Coding Agent's picks against the
    retrieved guideline text and the tool's candidate list, dropping any
    code that isn't actually supported.

If the crewai package isn't installed/importable, falls back to an
equivalent two-step workflow driven directly through our own Gemini LLM
client so the coding feature still works end to end.
"""
from __future__ import annotations

import json
import os

from agents.llm import DEFAULT_MODEL, get_llm_client
from agents.prompts import CODING_SCHEMA_HINT, CODING_SYSTEM_PROMPT_V1
from agents.schemas import CodingOutput, ICD10Suggestion
from retrieval.hybrid_search import RetrievedChunk, hybrid_search
from tools.icd10_lookup import search_icd10

VERIFIER_SYSTEM_PROMPT = """\
You are the Verification agent in a medical coding crew. Given a proposed
list of ICD-10 codes with rationale, and the original candidate codes
returned by the ICD-10 lookup tool plus the retrieved clinical context,
remove any suggested code that is not actually supported by the tool's
candidate list or the retrieved context. Keep confidence scores for codes
you keep; do not invent new codes.
"""

VERIFIER_SCHEMA_HINT = """\
Return JSON with exactly these keys:
{"suggested_codes": [{"code": string, "description": string, "confidence": number}], "rationale": string}
"""


def _gather_candidates(user_input: str, query: str, chunks: list[RetrievedChunk]) -> list[dict]:
    candidates = search_icd10(query, max_results=8)
    if not candidates:
        for chunk in chunks:
            candidates.extend(search_icd10(chunk.title, max_results=3))
    return candidates


# --- Fallback path: direct two-step workflow via our Gemini LLM client ---


def _coding_agent_task(user_input: str, chunks: list[RetrievedChunk], candidates: list[dict]) -> CodingOutput:
    llm = get_llm_client()
    context = "\n\n".join(f"[{c.title}] {c.text}" for c in chunks)
    candidates_block = "\n".join(f"- {c['code']}: {c['description']}" for c in candidates)
    user_prompt = (
        f"Clinical request:\n{user_input}\n\n"
        f"Candidate ICD-10 codes from lookup tool:\n{candidates_block}\n\n"
        f"Retrieved guideline context:\n{context}"
    )
    raw = llm.structured_completion(
        role="coding",
        system_prompt=CODING_SYSTEM_PROMPT_V1,
        user_prompt=user_prompt,
        schema_hint=CODING_SCHEMA_HINT,
    )
    try:
        return CodingOutput(**raw)
    except Exception:
        return CodingOutput(
            suggested_codes=[
                ICD10Suggestion(code=c["code"], description=c["description"], confidence=0.5)
                for c in candidates[:3]
            ],
            rationale="Fallback: LLM output could not be parsed; returning top lookup candidates.",
        )


def _verification_agent_task(
    coding_output: CodingOutput, chunks: list[RetrievedChunk], candidates: list[dict]
) -> CodingOutput:
    llm = get_llm_client()
    context = "\n\n".join(f"[{c.title}] {c.text}" for c in chunks)
    candidates_block = "\n".join(f"- {c['code']}: {c['description']}" for c in candidates)
    user_prompt = (
        f"Proposed codes:\n{[s.model_dump() for s in coding_output.suggested_codes]}\n\n"
        f"Rationale:\n{coding_output.rationale}\n\n"
        f"Tool candidate codes:\n{candidates_block}\n\n"
        f"Retrieved context:\n{context}"
    )
    raw = llm.structured_completion(
        role="coding",
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        schema_hint=VERIFIER_SCHEMA_HINT,
    )
    try:
        return CodingOutput(**raw)
    except Exception:
        valid_codes = {c["code"] for c in candidates}
        filtered = [s for s in coding_output.suggested_codes if s.code in valid_codes]
        return CodingOutput(suggested_codes=filtered, rationale=coding_output.rationale)


def _run_fallback(user_input: str, chunks: list[RetrievedChunk], candidates: list[dict]) -> CodingOutput:
    coding_output = _coding_agent_task(user_input, chunks, candidates)
    return _verification_agent_task(coding_output, chunks, candidates)


# --- Primary path: real CrewAI Agent/Task/Crew ---


def _run_crewai(user_input: str, chunks: list[RetrievedChunk], candidates: list[dict]) -> CodingOutput:
    from crewai import Agent, Crew, Process, Task
    from crewai.tools import tool

    @tool("ICD-10 Lookup")
    def icd10_lookup_tool(term: str) -> str:
        """Look up candidate ICD-10-CM codes for a clinical term."""
        return json.dumps(search_icd10(term, max_results=8))

    llm_model = f"gemini/{DEFAULT_MODEL}"
    context = "\n\n".join(f"[{c.title}] {c.text}" for c in chunks)
    candidates_block = "\n".join(f"- {c['code']}: {c['description']}" for c in candidates)

    coder = Agent(
        role="Medical Coding Agent",
        goal="Propose the most clinically appropriate ICD-10-CM codes for the clinical request.",
        backstory="An experienced clinical coder who only ever cites codes supported by tool lookups or guideline context.",
        tools=[icd10_lookup_tool],
        llm=llm_model,
        verbose=False,
    )
    verifier = Agent(
        role="Coding Verification Agent",
        goal="Cross-check proposed ICD-10 codes against the lookup tool and guideline context, dropping unsupported codes.",
        backstory="A meticulous auditor who removes any code not backed by evidence.",
        tools=[icd10_lookup_tool],
        llm=llm_model,
        verbose=False,
    )

    coding_task = Task(
        description=(
            f"Clinical request:\n{user_input}\n\n"
            f"Candidate ICD-10 codes from lookup tool:\n{candidates_block}\n\n"
            f"Retrieved guideline context:\n{context}\n\n"
            f"{CODING_SCHEMA_HINT}"
        ),
        expected_output="A JSON object with suggested_codes and rationale, as specified.",
        agent=coder,
    )
    verify_task = Task(
        description=(
            f"Verify the coding agent's proposed codes against the tool candidates and context below. "
            f"Tool candidate codes:\n{candidates_block}\n\nRetrieved context:\n{context}\n\n{VERIFIER_SCHEMA_HINT}"
        ),
        expected_output="A JSON object with suggested_codes and rationale, as specified.",
        agent=verifier,
        context=[coding_task],
    )

    crew = Crew(agents=[coder, verifier], tasks=[coding_task, verify_task], process=Process.sequential, verbose=False)
    result = crew.kickoff()

    raw_text = str(result.raw if hasattr(result, "raw") else result)
    try:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[4:] if text.lower().startswith("json") else text
        return CodingOutput(**json.loads(text.strip()))
    except Exception:
        valid_codes = {c["code"] for c in candidates}
        return CodingOutput(
            suggested_codes=[
                ICD10Suggestion(code=c["code"], description=c["description"], confidence=0.5)
                for c in candidates[:3]
                if c["code"] in valid_codes
            ],
            rationale="CrewAI output could not be parsed as JSON; returning top lookup candidates as a safe fallback.",
        )


def run_coding_crew(user_input: str, search_query: str | None = None) -> CodingOutput:
    """Runs the two-role coding crew end to end and returns the verified
    ICD-10 coding suggestions. Prefers the real CrewAI framework; falls
    back to a direct two-step Gemini workflow if crewai isn't available
    or raises at runtime (e.g. missing optional sub-dependency)."""
    query = search_query or user_input
    chunks = hybrid_search(query, k=5)
    candidates = _gather_candidates(user_input, query, chunks)

    if os.getenv("USE_FALLBACK_CODING_CREW") == "1":
        return _run_fallback(user_input, chunks, candidates)

    try:
        return _run_crewai(user_input, chunks, candidates)
    except Exception:
        return _run_fallback(user_input, chunks, candidates)
