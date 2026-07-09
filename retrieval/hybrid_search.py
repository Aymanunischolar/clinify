"""Hybrid retrieval: merges vector similarity (Chroma) and keyword BM25
(Elasticsearch) results with normalized score fusion, then reranks the
merged candidate set with a cross-encoder (falls back to the fused score
if no cross-encoder model is available offline).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from retrieval.chunking import Chunk, load_and_chunk_directory
from retrieval.elasticsearch_client import get_keyword_index
from retrieval.vector_store import get_vector_store

VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.6"))
KEYWORD_WEIGHT = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.4"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RetrievedChunk:
    id: str
    text: str
    title: str
    source: str
    vector_score: float
    keyword_score: float
    fused_score: float
    rerank_score: float | None = None


def _normalize(results: list[dict]) -> dict[str, float]:
    if not results:
        return {}
    scores = [r["score"] for r in results]
    lo, hi = min(scores), max(scores)
    span = (hi - lo) or 1.0
    return {r["id"]: (r["score"] - lo) / span for r in results}


def ingest_directory(directory: str = "data/sample_docs", chunk_size: int = 800, overlap: int = 120) -> int:
    path = Path(directory)
    if not path.is_absolute():
        # Resolve relative to the project root rather than the process's
        # CWD, which serverless runtimes (e.g. Vercel) don't guarantee.
        path = PROJECT_ROOT / path
    chunks = load_and_chunk_directory(path, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return 0
    vector_store = get_vector_store()
    keyword_index = get_keyword_index()
    vector_store.add_chunks(chunks)
    keyword_index.add_chunks(chunks)
    return len(chunks)


def reset_indexes():
    get_vector_store().reset()
    get_keyword_index().reset()


def ensure_ingested(directory: str = "data/sample_docs") -> None:
    """Ingests the sample docs if the vector store is currently empty.

    On serverless deployments (e.g. Vercel) nothing persists across cold
    starts, so callers that need the knowledge base populated should call
    this instead of assuming a prior /ingest call already ran.
    """
    if get_vector_store().count() == 0:
        ingest_directory(directory)


@lru_cache(maxsize=1)
def _get_reranker():
    try:
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception:
        return None


def rerank(query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
    reranker = _get_reranker()
    if reranker is None or not candidates:
        for c in candidates:
            c.rerank_score = c.fused_score
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
    pairs = [(query, c.text) for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c.rerank_score = float(s)
    return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)


def hybrid_search(query: str, k: int = 6, candidate_pool: int = 20) -> list[RetrievedChunk]:
    vector_store = get_vector_store()
    keyword_index = get_keyword_index()

    vector_hits = vector_store.similarity_search(query, k=candidate_pool)
    keyword_hits = keyword_index.search(query, k=candidate_pool)

    vector_norm = _normalize(vector_hits)
    keyword_norm = _normalize(keyword_hits)

    by_id: dict[str, dict] = {}
    for h in vector_hits:
        by_id[h["id"]] = h
    for h in keyword_hits:
        by_id.setdefault(h["id"], h)

    merged: list[RetrievedChunk] = []
    for cid, h in by_id.items():
        v = vector_norm.get(cid, 0.0)
        kw = keyword_norm.get(cid, 0.0)
        fused = VECTOR_WEIGHT * v + KEYWORD_WEIGHT * kw
        merged.append(
            RetrievedChunk(
                id=cid,
                text=h["text"],
                title=h.get("title", ""),
                source=h.get("source", ""),
                vector_score=v,
                keyword_score=kw,
                fused_score=fused,
            )
        )

    merged.sort(key=lambda c: c.fused_score, reverse=True)
    top_candidates = merged[: max(k * 3, candidate_pool)]
    reranked = rerank(query, top_candidates)
    return reranked[:k]
