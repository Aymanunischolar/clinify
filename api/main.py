"""FastAPI serving layer for ClinicalRAG Agent.

Endpoints:
  GET  /health              — liveness/readiness probe
  POST /ingest               — (re)ingest the sample knowledge base
  POST /query                — run the full agent graph, return a cited answer
  POST /query/stream         — same, but streams the writer's answer as SSE
  POST /coding                — run the CrewAI ICD-10 coding crew
  GET  /                     — serves the static chat UI
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("clinicalrag")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="ClinicalRAG Agent API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    correlation_id = request.headers.get("x-correlation-id", str(uuid.uuid4()))
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["x-correlation-id"] = correlation_id
    logger.info(
        json.dumps(
            {
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            }
        )
    )
    return response


class QueryRequest(BaseModel):
    query: str


class CodingRequest(BaseModel):
    query: str


class IngestRequest(BaseModel):
    directory: str = "data/sample_docs"
    reset: bool = False


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest):
    from retrieval.hybrid_search import ingest_directory, reset_indexes

    if req.reset:
        reset_indexes()
    count = ingest_directory(req.directory)
    return {"chunks_ingested": count}


@app.post("/query")
async def query(req: QueryRequest):
    from graph.build_graph import run_pipeline

    result = run_pipeline(req.query)
    return {
        "answer": result.get("final_answer", ""),
        "citations": result.get("citations", []),
        "key_findings": result.get("key_findings", []),
        "sub_questions": result.get("sub_questions", []),
        "requires_coding": result.get("requires_coding", False),
        "qa": result.get("qa_output", {}),
        "retrieved_chunks": [
            {
                "id": c.id,
                "title": c.title,
                "source": c.source,
                "fused_score": c.fused_score,
                "rerank_score": c.rerank_score,
            }
            for c in result.get("retrieved_chunks", [])
        ],
    }


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    from agents import planner, reasoner, retriever
    from agents.llm import get_llm_client
    from agents.prompts import WRITER_SYSTEM_PROMPT_V1
    from agents.writer import _format_context

    async def event_generator():
        plan = planner.run(req.query)
        yield f"event: plan\ndata: {json.dumps(plan.model_dump())}\n\n"

        chunks = retriever.run(plan.search_queries or [req.query])
        yield f"event: retrieval\ndata: {json.dumps([c.id for c in chunks])}\n\n"

        reasoning = reasoner.run(req.query, chunks)
        yield f"event: reasoning\ndata: {json.dumps(reasoning.model_dump())}\n\n"

        llm = get_llm_client()
        context = _format_context(chunks)
        findings_block = "\n".join(f"- {f}" for f in reasoning.key_findings)
        user_prompt = (
            f"Question:\n{req.query}\n\nReasoning summary:\n{findings_block}\n\n"
            f"Retrieved context (use for citations):\n{context}\n\n"
            "Respond with plain prose only (no JSON) for this streaming preview."
        )
        for token in llm.stream_completion(WRITER_SYSTEM_PROMPT_V1, user_prompt):
            yield f"event: token\ndata: {json.dumps(token)}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/coding")
async def coding(req: CodingRequest):
    from crew.coding_crew import run_coding_crew

    result = run_coding_crew(req.query)
    return result.model_dump()


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
