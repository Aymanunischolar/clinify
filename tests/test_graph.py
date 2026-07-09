from graph.build_graph import run_pipeline


def test_pipeline_produces_final_answer_and_citations():
    result = run_pipeline("What is first-line pharmacologic therapy for stage 2 hypertension?")
    assert result.get("final_answer")
    assert "citations" in result
    assert "qa_output" in result


def test_pipeline_retrieves_relevant_source():
    result = run_pipeline("What antibiotic is used for outpatient community-acquired pneumonia?")
    sources = [c.source for c in result.get("retrieved_chunks", [])]
    assert any("pneumonia" in s.lower() for s in sources)
