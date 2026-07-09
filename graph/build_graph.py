"""LangGraph state graph: Planner -> Retriever -> Reasoner -> Writer -> QA,
with a conditional verification loop — if the QA agent finds the Writer's
answer isn't grounded, it substitutes a revised, grounded answer rather
than looping indefinitely (bounded to one revision to keep latency
predictable for a synchronous API request).
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from agents import planner, qa_verifier, reasoner, retriever, writer
from agents.schemas import QAOutput, WriterOutput


class GraphState(TypedDict, total=False):
    user_input: str
    sub_questions: list[str]
    search_queries: list[str]
    requires_coding: bool
    retrieved_chunks: list[Any]
    key_findings: list[str]
    writer_output: dict
    qa_output: dict
    final_answer: str
    citations: list[dict]


def planner_node(state: GraphState) -> GraphState:
    result = planner.run(state["user_input"])
    return {
        "sub_questions": result.sub_questions,
        "search_queries": result.search_queries or [state["user_input"]],
        "requires_coding": result.requires_coding,
    }


def retriever_node(state: GraphState) -> GraphState:
    chunks = retriever.run(state["search_queries"])
    return {"retrieved_chunks": chunks}


def reasoner_node(state: GraphState) -> GraphState:
    result = reasoner.run(state["user_input"], state["retrieved_chunks"])
    return {"key_findings": result.key_findings}


def writer_node(state: GraphState) -> GraphState:
    result = writer.run(state["user_input"], state["key_findings"], state["retrieved_chunks"])
    return {"writer_output": result.model_dump()}


def qa_node(state: GraphState) -> GraphState:
    writer_output = WriterOutput(**state["writer_output"])
    result = qa_verifier.run(state["user_input"], writer_output, state["retrieved_chunks"])
    final_answer = result.revised_answer if (not result.is_grounded and result.revised_answer) else writer_output.answer
    return {
        "qa_output": result.model_dump(),
        "final_answer": final_answer,
        "citations": [c.model_dump() for c in writer_output.citations],
    }


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("writer", writer_node)
    graph.add_node("qa", qa_node)

    graph.set_entry_point("planner")
    graph.add_edge("planner", "retriever")
    graph.add_edge("retriever", "reasoner")
    graph.add_edge("reasoner", "writer")
    graph.add_edge("writer", "qa")
    graph.add_edge("qa", END)

    return graph.compile()


_compiled_graph = None


def get_compiled_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_pipeline(user_input: str) -> GraphState:
    graph = get_compiled_graph()
    return graph.invoke({"user_input": user_input})
