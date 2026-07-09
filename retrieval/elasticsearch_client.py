"""Elasticsearch BM25 keyword index, with an in-process BM25 fallback
(rank_bm25) when no Elasticsearch cluster is reachable. This lets the
hybrid retrieval pipeline run locally without docker-compose while still
being backed by a real ES mapping/query in the deployed configuration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from retrieval.chunking import Chunk

ES_URL = os.getenv("ELASTICSEARCH_URL", "http://localhost:9200")
ES_INDEX = os.getenv("ELASTICSEARCH_INDEX", "clinical_docs")

INDEX_MAPPING = {
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "english"},
            "title": {"type": "text"},
            "source": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
        }
    }
}


class ElasticsearchKeywordIndex:
    def __init__(self, url: str = ES_URL, index: str = ES_INDEX):
        from elasticsearch import Elasticsearch

        self.es = Elasticsearch(url)
        self.index = index
        if not self.es.indices.exists(index=self.index):
            self.es.indices.create(index=self.index, body=INDEX_MAPPING)

    def reset(self):
        if self.es.indices.exists(index=self.index):
            self.es.indices.delete(index=self.index)
        self.es.indices.create(index=self.index, body=INDEX_MAPPING)

    def add_chunks(self, chunks: list[Chunk]):
        from elasticsearch.helpers import bulk

        actions = [
            {
                "_index": self.index,
                "_id": c.id,
                "_source": {
                    "text": c.text,
                    "title": c.title,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                },
            }
            for c in chunks
        ]
        bulk(self.es, actions)
        self.es.indices.refresh(index=self.index)

    def search(self, query: str, k: int = 8) -> list[dict]:
        resp = self.es.search(
            index=self.index,
            query={"match": {"text": {"query": query}}},
            size=k,
        )
        hits = resp["hits"]["hits"]
        max_score = hits[0]["_score"] if hits else 1.0
        return [
            {
                "id": h["_id"],
                "text": h["_source"]["text"],
                "title": h["_source"]["title"],
                "source": h["_source"]["source"],
                "chunk_index": h["_source"]["chunk_index"],
                "score": h["_score"] / max_score if max_score else 0.0,
            }
            for h in hits
        ]

    def count(self) -> int:
        return self.es.count(index=self.index)["count"]


class LocalBM25Index:
    """rank_bm25-backed fallback that mirrors the ES index's behavior."""

    def __init__(self, persist_path: Path = Path(os.getenv("VECTOR_STORE_DIR", "data/vector_store")) / "bm25_docs.json"):
        self.persist_path = persist_path
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._docs: list[dict] = []
        if self.persist_path.exists():
            self._docs = json.loads(self.persist_path.read_text(encoding="utf-8"))
        self._bm25 = None
        self._build()

    def reset(self):
        self._docs = []
        self._bm25 = None
        self._save()

    def add_chunks(self, chunks: list[Chunk]):
        for c in chunks:
            self._docs.append(
                {
                    "id": c.id,
                    "text": c.text,
                    "title": c.title,
                    "source": c.source,
                    "chunk_index": c.chunk_index,
                }
            )
        self._save()
        self._build()

    def _save(self):
        self.persist_path.write_text(json.dumps(self._docs), encoding="utf-8")

    def _build(self):
        if not self._docs:
            self._bm25 = None
            return
        from rank_bm25 import BM25Okapi

        tokenized = [d["text"].lower().split() for d in self._docs]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 8) -> list[dict]:
        if not self._bm25 or not self._docs:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        max_score = max(scores) if len(scores) else 1.0
        ranked = sorted(zip(self._docs, scores), key=lambda x: x[1], reverse=True)[:k]
        return [
            {**doc, "score": (score / max_score) if max_score else 0.0}
            for doc, score in ranked
            if score > 0
        ]

    def count(self) -> int:
        return len(self._docs)


def get_keyword_index():
    if os.getenv("USE_LOCAL_BM25") == "1":
        return LocalBM25Index()
    try:
        return ElasticsearchKeywordIndex()
    except Exception:
        return LocalBM25Index()
