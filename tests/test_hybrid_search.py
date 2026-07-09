from retrieval.hybrid_search import hybrid_search


def test_hybrid_search_returns_relevant_chunk_for_hypertension_query():
    results = hybrid_search("first-line therapy for stage 2 hypertension", k=5)
    assert len(results) > 0
    assert any("hypertension" in r.source.lower() for r in results)


def test_hybrid_search_returns_relevant_chunk_for_diabetes_query():
    results = hybrid_search("metformin renal contraindication", k=5)
    assert len(results) > 0
    assert any("diabetes" in r.source.lower() for r in results)


def test_hybrid_search_respects_k():
    results = hybrid_search("pneumonia antibiotic therapy", k=2)
    assert len(results) <= 2


def test_fused_score_is_between_0_and_1():
    results = hybrid_search("CURB-65 score", k=5)
    for r in results:
        assert 0.0 <= r.fused_score <= 1.0 + 1e-6
