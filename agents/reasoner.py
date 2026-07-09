"""Reasoning agent node: synthesizes key findings from retrieved context."""
from __future__ import annotations

from agents.llm import get_llm_client
from agents.prompts import REASONER_SCHEMA_HINT, REASONER_SYSTEM_PROMPT_V1
from agents.schemas import ReasonerOutput
from retrieval.hybrid_search import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[chunk_id={c.id}] ({c.title} — {c.source})\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def run(user_question: str, chunks: list[RetrievedChunk]) -> ReasonerOutput:
    llm = get_llm_client()
    context = _format_context(chunks)
    user_prompt = f"Question:\n{user_question}\n\nRetrieved context:\n{context}"
    raw = llm.structured_completion(
        system_prompt=REASONER_SYSTEM_PROMPT_V1,
        user_prompt=user_prompt,
        schema_hint=REASONER_SCHEMA_HINT,
    )
    try:
        return ReasonerOutput(**raw)
    except Exception:
        return ReasonerOutput(key_findings=[], supporting_chunk_ids=[c.id for c in chunks])
