from agents import planner, reasoner, retriever, writer
from agents.qa_verifier import run as qa_run


def test_planner_returns_search_queries():
    result = planner.run("What is first-line therapy for hypertension?")
    assert len(result.search_queries) >= 1


def test_retriever_returns_chunks_for_planned_queries():
    chunks = retriever.run(["type 2 diabetes first-line agent"])
    assert len(chunks) > 0


def test_reasoner_produces_findings_from_context():
    chunks = retriever.run(["community-acquired pneumonia antibiotics"])
    result = reasoner.run("What antibiotic should be used for CAP?", chunks)
    assert isinstance(result.key_findings, list)


def test_writer_produces_answer_with_citations():
    chunks = retriever.run(["hypertension first-line therapy"])
    reasoning = reasoner.run("first-line hypertension therapy", chunks)
    result = writer.run("first-line hypertension therapy", reasoning.key_findings, chunks)
    assert result.answer
    assert isinstance(result.citations, list)


def test_qa_verifier_returns_grounded_verdict():
    chunks = retriever.run(["hypertension first-line therapy"])
    reasoning = reasoner.run("first-line hypertension therapy", chunks)
    writer_output = writer.run("first-line hypertension therapy", reasoning.key_findings, chunks)
    qa_output = qa_run("first-line hypertension therapy", writer_output, chunks)
    assert isinstance(qa_output.is_grounded, bool)
