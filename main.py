"""
HTTP RAG service.

Endpoints:
  POST /ingest         -> ingest all files in DATA_DIR (or a given path)
  POST /query          -> {question, top_k?, source_filter?} -> grounded answer
  GET  /health

Per-query latency, chunk count, and token usage are logged to stdout
(structured) as required.
"""
import time
import logging
import json
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from app.config import settings
from app.vectorstore import get_store
from app.llm import generate_answer
from app.ingest import ingest_directory, ingest_file

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rag")

app = FastAPI(title="Cost-Efficient RAG Service")


class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    source_filter: Optional[str] = None


class IngestRequest(BaseModel):
    path: Optional[str] = None  # file or falls back to DATA_DIR


@app.get("/health")
def health():
    store = get_store()
    return {"status": "ok", "vectors_stored": store.count()}


@app.post("/ingest")
def ingest(req: IngestRequest):
    if req.path and req.path.endswith((".pdf", ".html", ".htm", ".md", ".markdown", ".txt")):
        return ingest_file(req.path)
    return ingest_directory(req.path)


@app.post("/query")
def query(req: QueryRequest):
    top_k = req.top_k or settings.DEFAULT_TOP_K
    store = get_store()

    t0 = time.perf_counter()
    hits, retrieval_latency_ms = store.query(req.question, top_k, req.source_filter)

    # Filter out low-similarity hits -> avoid hallucinating from irrelevant context
    relevant_hits = [h for h in hits if h["similarity"] >= settings.MIN_SIMILARITY]

    gen = generate_answer(req.question, relevant_hits)
    total_latency_ms = (time.perf_counter() - t0) * 1000

    log_record = {
        "question": req.question,
        "top_k": top_k,
        "chunk_count_retrieved": len(hits),
        "chunk_count_used": len(relevant_hits),
        "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "input_tokens": gen["input_tokens"],
        "output_tokens": gen["output_tokens"],
        "mode": gen["mode"],
    }
    logger.info(json.dumps(log_record))

    return {
        "question": req.question,
        "answer": gen["answer"],
        "cited_chunk_ids": gen["cited_chunk_ids"],
        "retrieved_chunks": [
            {"id": h["id"], "source": h["metadata"]["source"], "similarity": round(h["similarity"], 4)}
            for h in hits
        ],
        "mode": gen["mode"],
        "latency_ms": round(total_latency_ms, 2),
        "tokens": {"input": gen["input_tokens"], "output": gen["output_tokens"]},
    }
