# ClinicalRAG Agent

A multi-agent, retrieval-augmented system for clinical documentation and knowledge
retrieval. Combines hybrid search (Elasticsearch BM25 + vector embeddings) with an
agentic orchestration layer built on **LangGraph** (primary pipeline) and **CrewAI**
(secondary ICD-10 coding crew), served through a FastAPI backend with a lightweight
web UI.

> Uses public, non-PHI sample clinical guideline text as its knowledge base. See
> [Design Decisions](#design-decisions--trade-offs) for what would change for real
> PHI/compliance requirements.

## Architecture

```
User Query / Transcript
        │
        ▼
   Planner Agent  ──►  Retriever Agent (Hybrid Search: BM25 + Vector + Rerank)
                              │
                              ▼
                        Reasoning Agent
                              │
                              ▼
                         Writer Agent
                              │
                              ▼
                    QA / Verification Agent
                              │
                              ▼
                  Final grounded answer + citations
```

Planner → Retriever → Reasoner → Writer → QA is a **LangGraph** state graph
([graph/build_graph.py](graph/build_graph.py)). A separate **CrewAI** crew
([crew/coding_crew.py](crew/coding_crew.py)) runs a Coding Agent + Verification Agent
for ICD-10 diagnostic coding suggestions, sharing the same hybrid retrieval layer and
demonstrating a second agentic framework plus tool-calling (ICD-10 lookup tool).

| Layer | Component | Technology |
|---|---|---|
| Ingestion | Chunking | Recursive paragraph/sentence splitter ([retrieval/chunking.py](retrieval/chunking.py)) |
| Retrieval | Vector store | Chroma (falls back to an in-process store if unavailable) |
| Retrieval | Keyword search | Elasticsearch BM25 (falls back to `rank_bm25` locally) |
| Retrieval | Fusion + reranking | Normalized score fusion + cross-encoder reranker |
| Orchestration | Primary graph | LangGraph: Planner → Retriever → Reasoner → Writer → QA |
| Orchestration | Secondary crew | CrewAI: Coding Agent + Verification Agent |
| LLM | Generation & reasoning | Google **Gemini** (`google-genai`), structured JSON outputs via Pydantic schemas |
| Tools | External integrations | ICD-10 lookup (NLM Clinical Tables), drug interaction check (NLM RxNav), FHIR formatter |
| Serving | API | FastAPI, async endpoints, SSE streaming, correlation-ID request logging |
| Frontend | UI | Static HTML/CSS/JS chat + coding + ingestion tabs, served by FastAPI |
| Evaluation | Quality gate | RAGAS-style proxy metrics (faithfulness, answer relevancy, context recall) over a golden query set |
| Deployment | Containers | Docker + docker-compose (API + Elasticsearch) |
| MLOps | CI/CD | GitHub Actions: lint → test → eval gate → build image |

## Runs without any API key

Every external dependency has a local fallback so the whole pipeline is demoable
offline:

- No `GEMINI_API_KEY` → LLM calls use a deterministic mock provider (clearly labeled
  `[MOCK MODE]` in output).
- No reachable Elasticsearch → falls back to an in-process `rank_bm25` index.
- No reachable Chroma → falls back to an in-process JSON-backed vector store.
- No `sentence-transformers` model available → falls back to a deterministic hashing
  embedder.
- No network for ICD-10 / drug-interaction lookups → falls back to small local tables.

Set `GEMINI_API_KEY` (see `.env.example`) to get real LLM-generated answers instead of
the mock template.

## Quickstart (local, no Docker)

There are two requirements files: `requirements.txt` (slim — no torch/chromadb/
elasticsearch/crewai, used by the Vercel deployment) and `requirements-full.txt`
(full-fidelity local stack: real sentence-transformers embeddings, cross-encoder
reranker, Elasticsearch client, CrewAI). Use the full one for local development.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements-full.txt
cp .env.example .env          # fill in GEMINI_API_KEY for real generation

uvicorn api.main:app --reload
```

Open http://localhost:8000 — the UI has three tabs:

1. **Knowledge Base** — click "Ingest Sample Docs" first to populate the vector +
   keyword indexes with the sample hypertension / type 2 diabetes / pneumonia
   guidelines.
2. **Ask a Question** — runs the full LangGraph pipeline and shows the pipeline trace,
   cited answer, QA verdict, and reasoning findings. Toggle "Stream response" for an
   SSE token stream.
3. **ICD-10 Coding** — runs the CrewAI coding crew and shows verified suggested codes.

## Quickstart (Docker)

```bash
docker compose -f docker/docker-compose.yml up --build
```

Runs the API against a real Elasticsearch container. `POST /ingest` (or the UI) to
populate the knowledge base.

## Deployment (Vercel)

The frontend and API deploy together as a single Vercel project (`vercel.json` routes
every path to the `api/index.py` ASGI function, which just imports the same FastAPI
`app` used locally).

Vercel's serverless functions are stateless (nothing persists across cold starts) and
size-capped, so this deployment mode intentionally trades fidelity for portability:

- Uses `requirements.txt` (the slim manifest at the project root — Vercel's Python
  builder only reads a root-level manifest) instead of `requirements-full.txt`.
- No torch/sentence-transformers/chromadb/elasticsearch/crewai — every place that uses
  them already lazy-imports with a fallback (hashing embedder, in-process BM25,
  in-memory vector store, direct-Gemini coding crew instead of the CrewAI framework).
- `/query`, `/query/stream`, and `/coding` call `ensure_ingested()` to re-populate the
  in-memory store from `data/sample_docs` on first request after a cold start, since
  nothing survives between invocations (`/tmp` is the only writable path).

```bash
npm i -g vercel
vercel link
vercel env add GEMINI_API_KEY production   # paste your key
vercel --prod
```

For the full-fidelity experience (real embeddings, cross-encoder reranking, a real
Elasticsearch cluster, the actual CrewAI framework), use Docker or a host that supports
long-running processes and persistent disk (Render, Railway, Fly.io, a VM) instead.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness probe |
| POST | `/ingest` | (Re)ingest sample docs into vector + keyword indexes |
| POST | `/query` | Run the full LangGraph pipeline, return cited answer + trace |
| POST | `/query/stream` | Same, streamed as Server-Sent Events |
| POST | `/coding` | Run the CrewAI ICD-10 coding crew |

## Evaluation

```bash
python eval/run_ragas_eval.py
```

Runs the golden query set ([eval/golden_set.json](eval/golden_set.json)) through the
full pipeline and scores context recall, answer relevancy, and faithfulness. Exits
non-zero (gating CI) if the aggregate score drops below `EVAL_PASS_THRESHOLD` (default
`0.6`).

## Tests

```bash
pytest -v
```

## Design Decisions & Trade-offs

**Why hybrid search instead of vector-only?** Vector similarity alone misses exact
matches on medical codes, drug names, and acronyms (e.g. "CURB-65", "I10") that a
clinician expects literal recall on. BM25 keyword search covers that precision case;
vector search covers semantic/paraphrased queries. Normalized score fusion plus a
cross-encoder rerank gets both.

**Why two agentic frameworks?** LangGraph drives the primary, deterministic
retrieve-reason-write-verify pipeline where explicit state and conditional edges matter.
CrewAI drives the secondary ICD-10 coding workflow, where a role-based
"propose → verify" crew maps naturally onto CrewAI's Agent/Task abstraction. Using both
demonstrates framework fluency rather than lock-in to one tool.

**How does the QA/verification agent reduce hallucination risk?** The Writer agent must
cite `chunk_id`s from retrieved context; the QA agent independently re-checks the answer
against the same source excerpts and can substitute a corrected, grounded
`revised_answer` before anything is returned to the caller.

**Why does everything have an offline fallback?** So the project can be evaluated,
demoed, and CI-tested without provisioning Elasticsearch, a vector DB, or an LLM API key
— while still using the real technology when those are available.

**What would change for real PHI/compliance requirements?** This project intentionally
uses public, non-PHI sample text. A production deployment handling real patient data
would need: encryption at rest/in transit, BAA-covered infra and LLM providers,
audit logging of every retrieval/generation event, de-identification before any
third-party API call, RBAC on the API layer, and a human-in-the-loop sign-off step
before any generated note is written back to a clinical record.

## Repository Structure

```
agents/       Agent nodes (planner, retriever, reasoner, writer, qa_verifier),
              LLM abstraction, prompts, structured-output schemas
crew/         CrewAI coding crew (ICD-10 suggestions)
graph/        LangGraph state graph definition
retrieval/    Chunking, embeddings, vector store, Elasticsearch/BM25, hybrid search
tools/        ICD-10 lookup, drug interaction check, FHIR formatter
api/          FastAPI app
frontend/     Static chat/coding/ingestion UI
eval/         Golden set + RAGAS-style eval harness
tests/        pytest suite
docker/       Dockerfile + docker-compose
.github/      CI workflow
```
