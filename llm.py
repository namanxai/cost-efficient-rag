"""
LLM abstraction for grounded generation.

Design: this module is the ONLY place that talks to a model provider.
- If ANTHROPIC_API_KEY (or OPENAI_API_KEY) is set, it calls the real API.
- If no key is present, it falls back to a deterministic EXTRACTIVE answer
  built directly from the retrieved chunks, clearly labeled as such.

This lets the whole service run end-to-end (and be evaluated) with zero
external dependency, while making it a one-line env change to swap in a
real generator/judge later. See README "Discussion" for why this tradeoff
was made under the time constraint.
"""
import os
import re
import httpx
from typing import List, Dict, Tuple

from app.config import settings

NO_CONTEXT_MSG = "I don't have relevant context to answer that question."


def _provider() -> str:
    if settings.LLM_PROVIDER != "auto":
        return settings.LLM_PROVIDER
    if settings.ANTHROPIC_API_KEY:
        return "anthropic"
    if settings.OPENAI_API_KEY:
        return "openai"
    return "none"


def _approx_tokens(text: str) -> int:
    # Rough heuristic (chars/4) used only when a real API token count isn't available.
    return max(1, len(text) // 4)


def _build_prompt(question: str, chunks: List[Dict]) -> str:
    context_block = "\n\n".join(
        f"[chunk:{c['id']}] {c['text']}" for c in chunks
    )
    return (
        "Answer the question using ONLY the context below. "
        "Cite the chunk id(s) you used in square brackets, e.g. [chunk:xyz]. "
        "If the context does not contain the answer, reply exactly: "
        f"\"{NO_CONTEXT_MSG}\"\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"
    )


def _call_anthropic(prompt: str) -> Tuple[str, int, int]:
    resp = httpx.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": settings.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": settings.LLM_MODEL,
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    return text.strip(), usage.get("input_tokens", _approx_tokens(prompt)), usage.get("output_tokens", _approx_tokens(text))


def _call_openai(prompt: str) -> Tuple[str, int, int]:
    resp = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
        json={
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 500,
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    return text.strip(), usage.get("prompt_tokens", _approx_tokens(prompt)), usage.get("completion_tokens", _approx_tokens(text))


def _extractive_fallback(question: str, chunks: List[Dict]) -> str:
    """
    No API key available: produce a grounded, citation-bearing answer by
    extracting the most relevant sentences from the top chunks instead of
    generating free text. This avoids hallucination entirely (it literally
    can't say anything not present in the retrieved text) but is lower
    quality than a real LLM -- flagged explicitly in the output.
    """
    q_words = set(w.lower() for w in re.findall(r"\w+", question) if len(w) > 3)
    best_sentences = []
    for c in chunks:
        sentences = re.split(r"(?<=[.!?])\s+", c["text"])
        for s in sentences:
            s_words = set(w.lower() for w in re.findall(r"\w+", s))
            overlap = len(q_words & s_words)
            if overlap > 0:
                best_sentences.append((overlap, s.strip(), c["id"]))
    if not best_sentences:
        return NO_CONTEXT_MSG
    best_sentences.sort(key=lambda x: -x[0])
    top = best_sentences[:3]
    answer = " ".join(f"{s} [chunk:{cid}]" for _, s, cid in top)
    return f"[extractive-fallback mode, no LLM key configured] {answer}"


def generate_answer(question: str, chunks: List[Dict]) -> Dict:
    """
    Returns dict: answer, cited_chunk_ids, input_tokens, output_tokens, mode
    """
    if not chunks:
        return {
            "answer": NO_CONTEXT_MSG,
            "cited_chunk_ids": [],
            "input_tokens": 0,
            "output_tokens": 0,
            "mode": "no_context",
        }

    provider = _provider()
    prompt = _build_prompt(question, chunks)

    if provider == "anthropic":
        text, in_tok, out_tok = _call_anthropic(prompt)
        mode = "llm:anthropic"
    elif provider == "openai":
        text, in_tok, out_tok = _call_openai(prompt)
        mode = "llm:openai"
    else:
        text = _extractive_fallback(question, chunks)
        in_tok, out_tok = _approx_tokens(prompt), _approx_tokens(text)
        mode = "extractive_fallback"

    cited = re.findall(r"\[chunk:([^\]]+)\]", text)
    return {
        "answer": text,
        "cited_chunk_ids": list(dict.fromkeys(cited)),  # dedupe, preserve order
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "mode": mode,
    }
