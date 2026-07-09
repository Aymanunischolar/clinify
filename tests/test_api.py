from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_ingest():
    res = client.post("/ingest", json={"directory": "data/sample_docs"})
    assert res.status_code == 200
    assert res.json()["chunks_ingested"] >= 0


def test_query_endpoint_returns_answer_and_citations():
    res = client.post("/query", json={"query": "First-line therapy for hypertension?"})
    assert res.status_code == 200
    body = res.json()
    assert "answer" in body
    assert "citations" in body


def test_coding_endpoint_returns_suggested_codes_key():
    res = client.post("/coding", json={"query": "Patient with essential hypertension"})
    assert res.status_code == 200
    assert "suggested_codes" in res.json()
