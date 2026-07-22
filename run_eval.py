"""
Evaluation harness: runs the fixed 15-30 question set end-to-end through the
live retrieval + generation stack and computes all three required layers:
  1. Retrieval quality (Recall@k / Hit Rate, MRR, nDCG@k, context precision)
  2. Answer quality (faithfulness, relevance, EM/F1 where gold answers exist)
  3. Cost & latency (p50 / p95 retrieval latency; token usage)

Usage: python -m eval.run_eval [--top-k 4]
Writes eval/eval_results.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from app.vectorstore import get_store
from app.llm import generate_answer, NO_CONTEXT_MSG
from app.config import settings
from eval.metrics import (
    recall_at_k, hit_rate, reciprocal_rank, ndcg_at_k, context_precision,
    exact_match, token_f1,
)
from eval.judge import judge_answer


def run(top_k: int = None):
    top_k = top_k or settings.DEFAULT_TOP_K
    store = get_store()

    questions = json.loads((Path(__file__).parent / "eval_questions.json").read_text())

    per_question = []
    retrieval_latencies = []
    total_input_tokens = 0
    total_output_tokens = 0

    for q in questions:
        t0 = time.perf_counter()
        hits, retrieval_latency_ms = store.query(q["question"], top_k)
        retrieval_latencies.append(retrieval_latency_ms)

        retrieved_sources = [h["metadata"]["source"] for h in hits]
        relevant_sources = q["relevant_sources"]
        total_relevant_chunks = store.count_chunks_for_sources(relevant_sources)

        relevant_hits = [h for h in hits if h["similarity"] >= settings.MIN_SIMILARITY]
        gen = generate_answer(q["question"], relevant_hits)
        total_input_tokens += gen["input_tokens"]
        total_output_tokens += gen["output_tokens"]

        r_at_k = recall_at_k(retrieved_sources, relevant_sources)
        hr = hit_rate(retrieved_sources, relevant_sources)
        mrr = reciprocal_rank(retrieved_sources, relevant_sources)
        ndcg = ndcg_at_k(retrieved_sources, relevant_sources, top_k, total_relevant_chunks)
        ctx_prec = context_precision(retrieved_sources, relevant_sources)

        is_unanswerable = (q["gold_answer"] == "NOT_ANSWERABLE")
        correctly_abstained = is_unanswerable and (gen["answer"].strip() == NO_CONTEXT_MSG or len(relevant_hits) == 0)

        em = f1 = None
        if not is_unanswerable:
            em = exact_match(gen["answer"], q["gold_answer"])
            f1 = token_f1(gen["answer"], q["gold_answer"])

        judge = judge_answer(q["question"], gen["answer"], [h["text"] for h in relevant_hits])

        per_question.append({
            "id": q["id"],
            "question": q["question"],
            "retrieved_sources": retrieved_sources,
            "relevant_sources": relevant_sources,
            "recall_at_k": round(r_at_k, 3),
            "hit_rate": hr,
            "reciprocal_rank": round(mrr, 3),
            "ndcg_at_k": round(ndcg, 3),
            "context_precision": round(ctx_prec, 3),
            "answer": gen["answer"],
            "answer_mode": gen["mode"],
            "is_unanswerable_case": is_unanswerable,
            "correctly_abstained": correctly_abstained if is_unanswerable else None,
            "exact_match": em,
            "f1": round(f1, 3) if f1 is not None else None,
            "faithfulness": judge["faithfulness"],
            "relevance": judge["relevance"],
            "judge_mode": judge["mode"],
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
        })

    n = len(per_question)
    answerable = [p for p in per_question if not p["is_unanswerable_case"]]
    unanswerable = [p for p in per_question if p["is_unanswerable_case"]]

    summary = {
        "config": {
            "top_k": top_k,
            "num_questions": n,
            "chunk_size": settings.CHUNK_SIZE,
            "chunk_overlap": settings.CHUNK_OVERLAP,
            "embedding_note": "local hashing-tfidf (sandbox substitute; see README)",
        },
        "retrieval": {
            "mean_recall_at_k": round(statistics.mean(p["recall_at_k"] for p in per_question), 3),
            "hit_rate": round(statistics.mean(p["hit_rate"] for p in per_question), 3),
            "mrr": round(statistics.mean(p["reciprocal_rank"] for p in per_question), 3),
            "mean_ndcg_at_k": round(statistics.mean(p["ndcg_at_k"] for p in per_question), 3),
            "mean_context_precision": round(statistics.mean(p["context_precision"] for p in answerable), 3) if answerable else None,
        },
        "answer_quality": {
            "mean_faithfulness": round(statistics.mean(p["faithfulness"] for p in per_question), 3),
            "mean_relevance": round(statistics.mean(p["relevance"] for p in per_question), 3),
            "mean_exact_match": round(statistics.mean(p["exact_match"] for p in answerable), 3) if answerable else None,
            "mean_f1": round(statistics.mean(p["f1"] for p in answerable), 3) if answerable else None,
            "abstention_accuracy": (
                round(sum(1 for p in unanswerable if p["correctly_abstained"]) / len(unanswerable), 3)
                if unanswerable else None
            ),
            "judge_mode": per_question[0]["judge_mode"] if per_question else None,
        },
        "latency_and_cost": {
            "p50_retrieval_latency_ms": round(statistics.median(retrieval_latencies), 3),
            "p95_retrieval_latency_ms": round(
                statistics.quantiles(retrieval_latencies, n=20)[18] if len(retrieval_latencies) >= 20
                else max(retrieval_latencies), 3
            ),
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "avg_tokens_per_query": round((total_input_tokens + total_output_tokens) / n, 1),
            "generation_mode": per_question[0]["answer_mode"] if per_question else None,
        },
    }

    output = {"summary": summary, "per_question": per_question}
    out_path = Path(__file__).parent / "eval_results.json"
    out_path.write_text(json.dumps(output, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nFull results written to {out_path}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()
    run(top_k=args.top_k)
