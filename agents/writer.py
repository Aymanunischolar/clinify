"""Writer agent node: drafts a cited, clinically precise answer/note."""
from __future__ import annotations

from agents.llm import get_llm_client
from agents.prompts import WRITER_SCHEMA_HINT, WRITER_SYSTEM_PROMPT_V1
from agents.schemas import WriterOutput
from retrieval.hybrid_search import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[chunk_id={c.id}] title={c.title!r} source={c.source!r}\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def run(user_question: str, key_findings: list[str], chunks: list[RetrievedChunk]) -> WriterOutput:
    llm = get_llm_client()
    context = _format_context(chunks)
    findings_block = "\n".join(f"- {f}" for f in key_findings)
    user_prompt = (
        f"Question:\n{user_question}\n\n"
        f"Reasoning summary:\n{findings_block}\n\n"
        f"Retrieved context (use for citations):\n{context}"
    )
    raw = llm.structured_completion(
        role="writer",
        system_prompt=WRITER_SYSTEM_PROMPT_V1,
        user_prompt=user_prompt,
        schema_hint=WRITER_SCHEMA_HINT,
    )
    try:
        return WriterOutput(**raw)
    except Exception:
        fallback_citations = [
            {"chunk_id": c.id, "title": c.title, "source": c.source} for c in chunks[:3]
        ]
        return WriterOutput(
            answer=raw.get("answer", "Unable to generate a grounded answer.") if isinstance(raw, dict) else str(raw),
            citations=fallback_citations,
        )
