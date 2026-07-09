"""Retriever agent node: runs hybrid search for each planned query and
merges/deduplicates the results into a single candidate context set.
"""
from __future__ import annotations

from retrieval.hybrid_search import RetrievedChunk, hybrid_search


def run(search_queries: list[str], k_per_query: int = 5, k_total: int = 8) -> list[RetrievedChunk]:
    seen: dict[str, RetrievedChunk] = {}
    for query in search_queries:
        for chunk in hybrid_search(query, k=k_per_query):
            existing = seen.get(chunk.id)
            if existing is None or chunk.fused_score > existing.fused_score:
                seen[chunk.id] = chunk
    ranked = sorted(seen.values(), key=lambda c: (c.rerank_score or c.fused_score), reverse=True)
    return ranked[:k_total]
