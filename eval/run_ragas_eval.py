"""RAGAS-style evaluation harness, gated in CI.

Runs the golden query set through the full agent pipeline and computes:
  - context_recall_proxy: did retrieval pull at least one chunk from the
    expected source document for each query?
  - answer_relevancy_proxy: fraction of expected keyword phrases present
    in the final answer (case-insensitive substring match).
  - faithfulness: pass-through of the QA agent's is_grounded verdict.

These are lightweight, deterministic proxies for the RAGAS metrics named
in the project design (faithfulness, answer relevancy, context
precision/recall). If the `ragas` package plus a compatible LLM/embeddings
wrapper for Gemini is wired in later, this script is the natural place to
swap in the real RAGAS `evaluate()` call — the golden set and pipeline
plumbing here would not need to change.

Exit code is non-zero if the aggregate score is below EVAL_PASS_THRESHOLD,
so this can gate a CI job.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

EVAL_PASS_THRESHOLD = float(os.getenv("EVAL_PASS_THRESHOLD", "0.6"))
GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def evaluate_one(case: dict) -> dict:
    from graph.build_graph import run_pipeline

    result = run_pipeline(case["query"])
    answer = (result.get("final_answer") or "").lower()
    sources = [c.source for c in result.get("retrieved_chunks", [])]

    context_recall = any(
        any(expected in src for src in sources) for expected in case["expected_sources"]
    )

    keyword_hits = sum(1 for kw in case["expected_answer_contains"] if kw.lower() in answer)
    answer_relevancy = keyword_hits / max(len(case["expected_answer_contains"]), 1)

    faithfulness = bool(result.get("qa_output", {}).get("is_grounded", True))

    case_score = (int(context_recall) + answer_relevancy + int(faithfulness)) / 3
    return {
        "id": case["id"],
        "query": case["query"],
        "context_recall_proxy": context_recall,
        "answer_relevancy_proxy": round(answer_relevancy, 2),
        "faithfulness": faithfulness,
        "case_score": round(case_score, 2),
    }


def main() -> int:
    golden_set = json.loads(GOLDEN_SET_PATH.read_text(encoding="utf-8"))

    from retrieval.hybrid_search import ingest_directory

    ingest_directory("data/sample_docs")

    results = [evaluate_one(case) for case in golden_set]
    aggregate = sum(r["case_score"] for r in results) / len(results)

    print(json.dumps({"results": results, "aggregate_score": round(aggregate, 3)}, indent=2))

    if aggregate < EVAL_PASS_THRESHOLD:
        print(
            f"\nEVAL FAILED: aggregate score {aggregate:.3f} is below threshold {EVAL_PASS_THRESHOLD}",
            file=sys.stderr,
        )
        return 1

    print(f"\nEVAL PASSED: aggregate score {aggregate:.3f} >= threshold {EVAL_PASS_THRESHOLD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
