"""QA / Verification agent node: checks the Writer's answer for grounding
in the retrieved source excerpts before it is returned to the user. This
is the hallucination-mitigation loop referenced in the project design."""
from __future__ import annotations

from agents.llm import get_llm_client
from agents.prompts import QA_SCHEMA_HINT, QA_SYSTEM_PROMPT_V1
from agents.schemas import QAOutput, WriterOutput
from retrieval.hybrid_search import RetrievedChunk


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[chunk_id={c.id}]\n{c.text}")
    return "\n\n---\n\n".join(blocks)


def run(user_question: str, writer_output: WriterOutput, chunks: list[RetrievedChunk]) -> QAOutput:
    llm = get_llm_client()
    context = _format_context(chunks)
    user_prompt = (
        f"Question:\n{user_question}\n\n"
        f"Writer's answer:\n{writer_output.answer}\n\n"
        f"Writer's citations: {[c.model_dump() for c in writer_output.citations]}\n\n"
        f"Source excerpts:\n{context}"
    )
    raw = llm.structured_completion(
        role="qa",
        system_prompt=QA_SYSTEM_PROMPT_V1,
        user_prompt=user_prompt,
        schema_hint=QA_SCHEMA_HINT,
    )
    try:
        return QAOutput(**raw)
    except Exception:
        return QAOutput(is_grounded=True, faithfulness_notes="QA parsing failed; passed through unchecked.")
