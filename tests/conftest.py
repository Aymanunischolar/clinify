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

# Tests must be deterministic and must not depend on a live LLM quota, so
# they always run against the mock LLM provider regardless of whether a
# real GEMINI_API_KEY is set (directly, or via api.main's load_dotenv()
# picking up .env at import time).
os.environ["FORCE_MOCK_LLM"] = "1"

import pytest


@pytest.fixture(autouse=True, scope="session")
def _ingest_sample_docs():
    from retrieval.hybrid_search import ingest_directory, reset_indexes

    reset_indexes()
    ingest_directory("data/sample_docs")
    yield
