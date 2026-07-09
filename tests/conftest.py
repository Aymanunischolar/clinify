import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("USE_IN_MEMORY_STORE", "1")
os.environ.setdefault("USE_LOCAL_BM25", "1")
os.environ.setdefault("USE_HASHING_EMBEDDER", "1")
os.environ.setdefault("USE_LOCAL_ICD10", "1")
os.environ.setdefault("USE_LOCAL_DRUG_CHECK", "1")
os.environ.setdefault("USE_FALLBACK_CODING_CREW", "1")
os.environ.setdefault("VECTOR_STORE_DIR", "data/test_vector_store")

import pytest


@pytest.fixture(autouse=True, scope="session")
def _ingest_sample_docs():
    from retrieval.hybrid_search import ingest_directory, reset_indexes

    reset_indexes()
    ingest_directory("data/sample_docs")
    yield
