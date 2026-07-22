# Cost-Efficient RAG Service

A QA service over a small document corpus, backed by ChromaDB (embedded, no
always-on server pods), with a full retrieval + answer evaluation harness and
a cost comparison against a managed vector DB.

## Vector store choice: ChromaDB

Chosen over pgvector / Qdrant / LanceDB / FAISS / sqlite-vec because:
- **Zero infra to stand up** — no Postgres instance, no separate server process
  or docker daemon requirement. Embedded mode is a real, persistent on-disk
  ANN index, not a toy — this made it the fastest path to something genuinely
  *runnable*, not just architecturally described.
- **Built-in metadata filtering** (`where={"source": ...}`) out of the box,
  satisfying the "at least one metadata filter" requirement without extra code.
- **Cosine similarity + persistence** on local disk, which is exactly the cost
  contrast the assignment is asking for: pay for bytes stored, not for an
  always-on managed pod.

FAISS would be the leaner alternative (even less overhead, no metadata store of
its own — you'd pair it with SQLite for metadata) and is the natural "bonus"
benchmark if there were more time.

## Important environment note (read this first)

This was built inside a network-sandboxed environment for the exercise:

1. **Embeddings**: ChromaDB's default embedding function downloads an ONNX
   MiniLM model from a host outside the sandbox's allowed egress list, and
   installing `sentence-transformers` (which pulls in PyTorch, multi-GB) hit a
   disk quota. So `app/local_embeddings.py` implements a **local, network-free
   TF-IDF-weighted feature-hashing embedding** (scikit-learn `HashingVectorizer`
   + `TfidfTransformer`, 384-dim, L2-normalized, real cosine similarity search).
   This captures **lexical/keyword** similarity well but not deep semantic
   similarity (no synonym understanding). **In a normal environment, swap this
   for `sentence-transformers("all-MiniLM-L6-v2")` or an API embedding model**
   — the interface (`app/vectorstore.py`) is already written against Chroma's
   standard `EmbeddingFunction` protocol, so it's a one-class swap.
2. **Generation/judging**: No LLM API key was available for this build.
   `app/llm.py` calls a real Anthropic/OpenAI endpoint if `ANTHROPIC_API_KEY` /
   `OPENAI_API_KEY` is set, and otherwise falls back to a deterministic
   **extractive** answer (top-matching sentences from retrieved chunks, always
   cited, never hallucinated) so the service is honestly runnable end-to-end
   with zero external dependencies. `eval/judge.py` mirrors this: real
   LLM-as-judge if a key is present, lexical-overlap heuristic otherwise. Every
   result in `eval/eval_results.json` is labeled with which mode produced it
   (`extractive_fallback` / `llm:anthropic` / `heuristic_fallback` / etc.) —
   nothing is silently faked as a "real" LLM result.

Both fallbacks exist so the **whole pipeline — ingest, retrieve, generate,
evaluate — is runnable and its numbers are real**, not so the design looks
finished. Flip two env vars and it's a normal LLM-backed RAG service.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY / OPENAI_API_KEY here if you have one
```

## Ingest

```bash
python -m app.ingest              # ingests everything in data/corpus/
```
Idempotent: re-running does not create duplicate vectors. Each chunk's ID is
`{source}::{chunk_index}::{sha256(chunk_text)[:16]}` — re-ingesting identical
content upserts the same IDs rather than appending new ones (verified: vector
count stays constant across repeated ingest runs).

## Run the service

```bash
uvicorn app.main:app --reload
```
- `GET /health` → `{"status": "ok", "vectors_stored": N}`
- `POST /ingest` → `{"path": "optional/specific/file.md"}` (omit to re-ingest `DATA_DIR`)
- `POST /query` → `{"question": "...", "top_k": 4, "source_filter": "mars_exploration.md"}`

Every query logs a structured JSON line (question, top_k, chunk counts,
retrieval + total latency, input/output tokens, generation mode) to stdout.

## Evaluate

```bash
python -m eval.run_eval --top-k 4
```
Runs all 20 fixed questions in `eval/eval_questions.json` through the live
stack and writes `eval/eval_results.json` (summary + full per-question detail).

### Current results (extractive-fallback mode, local hashing embeddings)

| Layer | Metric | Value |
|---|---|---|
| Retrieval | Recall@k / Hit Rate | 1.00 |
| Retrieval | MRR | 0.81 |
| Retrieval | nDCG@k | 0.75 |
| Retrieval | Context precision | 0.41 |
| Answer | Faithfulness (heuristic) | 0.85 |
| Answer | Relevance (heuristic) | 0.63 |
| Answer | Exact Match | 0.00 |
| Answer | F1 | 0.09 |
| Answer | Abstention accuracy (unanswerable case) | 0.00 |
| Latency | p50 / p95 retrieval | 1.5ms / 6.2ms |
| Cost | Avg tokens/query | ~1,138 |

See `cost_comparison.md` for the storage/pod cost table at 100K/1M/10M vectors.

## Discussion

**Was retrieval or generation the weak link?** Retrieval, clearly, in this
build — and the eval caught a concrete, honest example of it: the deliberately
unanswerable question *"What is the boiling point of gold used in spacecraft
manufacturing?"* retrieved the JWST chunk because it mentions "gold-coated
beryllium," a pure keyword collision. The system then answered instead of
abstaining (`abstention_accuracy: 0.0` on this one case). This is a direct
consequence of the lexical-hashing embedding substitution — a real semantic
embedding model would very likely separate "boiling point of gold" from
"gold-coated mirror segments" by meaning rather than by the shared token
"gold." Context precision (0.41) is also mediocre for the same reason: half of
retrieved chunks, on average, aren't actually relevant to the question, even
though *something* relevant is almost always in the top-k (hit rate 1.0).
Generation, by contrast, is deliberately hallucination-proof in fallback mode
(it can only ever say things that are literally present in retrieved text) —
so its main quality ceiling is fluency and conciseness (reflected in the near-
zero EM and low F1: it returns full grounding sentences, not a terse gold-style
answer), not correctness. **With a real LLM swapped in for both embeddings and
generation, I'd expect the retrieval numbers to improve substantially and the
generation numbers (EM/F1, relevance) to improve as well, while faithfulness
could either stay high or drop slightly** — a real generator is free to
paraphrase or (if not carefully prompted) drift from the source, which the
extractive fallback structurally cannot do.

**When would I switch back to a managed vector DB?** Three triggers, in order
of how often I'd expect to hit them in practice: (1) **query volume** — once
you're serving enough sustained QPS that you need real replica scaling and
low-latency guarantees under load, the ops burden of running that yourself
starts costing more engineer-hours than the managed premium; (2) **corpus
size + recall requirements** — past tens of millions of vectors, self-hosted
ANN tuning (HNSW parameters, sharding) becomes a real specialty, and a managed
service's tuned defaults start looking cheap by comparison; (3) **compliance /
availability requirements** — if the product needs multi-region HA, audit
logging, or a SOC2 attestation, buying that off-the-shelf from a managed
vendor is usually cheaper than building and maintaining it. Below all three of
those thresholds — which is most early-to-mid-stage products with a
lightly-queried index, exactly the scenario this assignment describes — the
self-hosted route is the credible, cheaper default this exercise set out to
prove.

## Project structure

```
app/
  config.py           # env-based settings, no hardcoded secrets
  loaders.py          # PDF / HTML / MD -> text
  chunking.py         # configurable size+overlap, deterministic chunk IDs
  local_embeddings.py # sandbox-safe embedding function (see note above)
  vectorstore.py       # ChromaDB wrapper: upsert, query, metadata filter
  llm.py              # generation: real API or extractive fallback
  ingest.py           # ingestion pipeline (file / directory)
  main.py             # FastAPI HTTP service + structured logging
eval/
  eval_questions.json # 20 fixed Q&A with doc-level relevance labels
  metrics.py           # Recall@k, Hit Rate, MRR, nDCG@k, context precision, EM/F1
  judge.py             # LLM-as-judge (faithfulness+relevance) or heuristic fallback
  run_eval.py          # ties it together, writes eval_results.json
data/corpus/           # sample corpus (3 short space-exploration docs)
cost_comparison.md     # cost table + assumptions
```
