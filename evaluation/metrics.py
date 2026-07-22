"""
Retrieval (IR) and answer-quality metrics.

Relevance judgments: for this eval set, a retrieved chunk is judged "relevant"
if its source document is in the question's `relevant_sources` label (doc-level
relevance proxy -- stated explicitly since the assignment requires "a notion of
which chunk is relevant" and our tiny corpus only has 2 chunks/doc, so doc-level
granularity is a defensible substitute for full chunk-level labeling. In a larger
corpus this should be chunk-level.)
"""
import math
import re
from typing import List, Dict


def is_relevant(chunk_source: str, relevant_sources: List[str]) -> bool:
    return chunk_source in relevant_sources


def recall_at_k(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """Recall@k: fraction of relevant items that were retrieved (also = Hit Rate
    when there's at least one relevant item, since we only need to find ANY)."""
    if not relevant_sources:
        return 1.0 if not retrieved_sources else 1.0  # nothing to recall, trivially satisfied
    hits = sum(1 for s in set(retrieved_sources) if s in relevant_sources)
    return hits / len(set(relevant_sources))


def hit_rate(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """1 if at least one relevant chunk was retrieved, else 0."""
    if not relevant_sources:
        return 1.0
    return 1.0 if any(s in relevant_sources for s in retrieved_sources) else 0.0


def reciprocal_rank(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    if not relevant_sources:
        return 1.0
    for rank, s in enumerate(retrieved_sources, start=1):
        if s in relevant_sources:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_sources: List[str], relevant_sources: List[str], k: int,
              total_relevant_chunks: int = None) -> float:
    """
    Binary relevance nDCG@k.

    `total_relevant_chunks`: the true count of relevant chunks in the corpus
    (not just relevant docs). Relevance here is judged at doc granularity, but
    a doc can contribute multiple chunks, so the ideal ranking's normalizer
    (IDCG) must use the real chunk count, not len(relevant_sources), or nDCG
    can exceed 1.0. If not provided, falls back to len(relevant_sources)
    (correct only when each relevant doc contributes exactly one chunk).
    """
    if not relevant_sources:
        return 1.0
    if total_relevant_chunks is None:
        total_relevant_chunks = len(relevant_sources)
    gains = [1.0 if s in relevant_sources else 0.0 for s in retrieved_sources[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal_gains = [1.0] * min(total_relevant_chunks, k)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def context_precision(retrieved_sources: List[str], relevant_sources: List[str]) -> float:
    """Fraction of retrieved chunks that are actually relevant (precision, not recall)."""
    if not retrieved_sources:
        return 0.0
    if not relevant_sources:
        # unanswerable question: precision is 1.0 only if nothing was (wrongly) retrieved
        # as confidently relevant; here we score on whether the system correctly
        # abstained downstream (handled separately in run_eval "no context" check).
        return 0.0 if retrieved_sources else 1.0
    relevant_count = sum(1 for s in retrieved_sources if s in relevant_sources)
    return relevant_count / len(retrieved_sources)


# ---------------- Answer-quality metrics ----------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize(pred).split()
    gold_tokens = normalize(gold).split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    gold_counts = {}
    for t in gold_tokens:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        overlap += min(c, common.get(t, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def groundedness_heuristic(answer: str, retrieved_chunks_text: List[str]) -> float:
    """
    Fallback faithfulness proxy when no LLM judge is available: fraction of
    the answer's content words that appear somewhere in the retrieved context.
    Used only in extractive_fallback mode; when a real LLM key is configured,
    run_eval.py instead calls the LLM as a judge (see judge_groundedness).
    """
    ans_words = set(normalize(answer).split())
    ctx_words = set()
    for c in retrieved_chunks_text:
        ctx_words |= set(normalize(c).split())
    ans_words = {w for w in ans_words if len(w) > 3 and w != "chunk"}
    if not ans_words:
        return 1.0
    grounded = sum(1 for w in ans_words if w in ctx_words)
    return grounded / len(ans_words)
