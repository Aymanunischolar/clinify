"""Vector store abstraction. Uses Chroma when available, otherwise an
in-memory numpy-free store so the system runs without extra services.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from retrieval.chunking import Chunk
from retrieval.embeddings import cosine_similarity, get_embedder

PERSIST_DIR = Path(os.getenv("VECTOR_STORE_DIR", "data/vector_store"))


class InMemoryVectorStore:
    def __init__(self, persist_path: Path = PERSIST_DIR / "memory_store.json"):
        self.persist_path = persist_path
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict] = []
        if self.persist_path.exists():
            self._records = json.loads(self.persist_path.read_text(encoding="utf-8"))

    def reset(self):
        self._records = []
        self._save()

    def add_chunks(self, chunks: list[Chunk]):
        embedder = get_embedder()
        vectors = embedder.embed([c.text for c in chunks])
        for chunk, vec in zip(chunks, vectors):
            self._records.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "title": chunk.title,
                    "chunk_index": chunk.chunk_index,
                    "metadata": chunk.metadata,
                    "embedding": vec,
                }
            )
        self._save()

    def _save(self):
        self.persist_path.write_text(json.dumps(self._records), encoding="utf-8")

    def similarity_search(self, query: str, k: int = 8) -> list[dict]:
        if not self._records:
            return []
        embedder = get_embedder()
        qvec = embedder.embed_one(query)
        scored = [
            {**{k2: v for k2, v in r.items() if k2 != "embedding"}, "score": cosine_similarity(qvec, r["embedding"])}
            for r in self._records
        ]
        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:k]

    def count(self) -> int:
        return len(self._records)


class ChromaVectorStore:
    def __init__(self, collection_name: str = "clinical_docs"):
        import chromadb

        self.client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        self.collection = self.client.get_or_create_collection(collection_name)

    def reset(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection("clinical_docs")

    def add_chunks(self, chunks: list[Chunk]):
        embedder = get_embedder()
        vectors = embedder.embed([c.text for c in chunks])
        self.collection.add(
            ids=[c.id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "title": c.title, "chunk_index": c.chunk_index, **c.metadata}
                for c in chunks
            ],
        )

    def similarity_search(self, query: str, k: int = 8) -> list[dict]:
        embedder = get_embedder()
        qvec = embedder.embed_one(query)
        res = self.collection.query(query_embeddings=[qvec], n_results=k)
        out = []
        if not res["ids"] or not res["ids"][0]:
            return out
        for i in range(len(res["ids"][0])):
            dist = res["distances"][0][i] if res.get("distances") else 0.0
            out.append(
                {
                    "id": res["ids"][0][i],
                    "text": res["documents"][0][i],
                    "score": 1.0 - dist,
                    **res["metadatas"][0][i],
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()


def get_vector_store():
    if os.getenv("USE_IN_MEMORY_STORE") == "1":
        return InMemoryVectorStore()
    try:
        return ChromaVectorStore()
    except Exception:
        return InMemoryVectorStore()
