"""Embedding provider abstraction.

Uses sentence-transformers when available. Falls back to a deterministic
hashing-based embedding so the pipeline still runs end-to-end offline /
without model downloads (useful for CI and quick demos).
"""
from __future__ import annotations

import hashlib
import math
import os
from functools import lru_cache

EMBED_DIM = 384


class Embedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class HashingEmbedder(Embedder):
    """Deterministic fallback embedder: token-hash bag-of-words projected
    into a fixed-size vector. Not semantically strong, but stable, fast,
    and dependency-free so the rest of the system (chunking, indexing,
    hybrid scoring, reranking, API) can be exercised without a model
    download.
    """

    def __init__(self, dim: int = EMBED_DIM):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h // self.dim) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if os.getenv("USE_HASHING_EMBEDDER") == "1":
        return HashingEmbedder()
    try:
        return SentenceTransformerEmbedder()
    except Exception:
        return HashingEmbedder()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(y * y for y in b)) or 1.0
    return dot / (na * nb)
