"""
Answer-quality judging: faithfulness/groundedness + answer relevance.

Uses the same LLM abstraction as generation (app.llm) so that when a real
API key is configured, this becomes genuine LLM-as-judge scoring. Falls
back to the lexical-overlap heuristic in metrics.py when no key is present,
so the eval harness always produces numbers, honestly labeled either way.
"""
import json
import re
from typing import Dict, List

from app.config import settings
from app.llm import _provider, _call_anthropic, _call_openai
from eval.metrics import groundedness_heuristic


JUDGE_PROMPT_TEMPLATE = """You are an evaluation judge. Score the ANSWER to the QUESTION using ONLY the CONTEXT provided.

Return ONLY a JSON object, no other text, with this exact shape:
{{"faithfulness": <0-1 float, is every claim in the answer supported by the context>, "relevance": <0-1 float, does the answer actually address the question>, "rationale": "<one sentence>"}}

CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}

JSON:"""


def _parse_judge_json(raw: str) -> Dict:
    """Robust parsing: strip code fences, find the first {...} block."""
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw[:200]}")
    obj = json.loads(match.group(0))
    obj["faithfulness"] = float(obj.get("faithfulness", 0.0))
    obj["relevance"] = float(obj.get("relevance", 0.0))
    return obj


def judge_answer(question: str, answer: str, context_chunks: List[str]) -> Dict:
    provider = _provider()
    context = "\n\n".join(context_chunks)

    if provider == "none":
        # Heuristic fallback -- documented clearly in the results file.
        faithfulness = groundedness_heuristic(answer, context_chunks)
        # crude relevance proxy: word overlap between question and answer
        q_words = set(re.findall(r"\w+", question.lower()))
        a_words = set(re.findall(r"\w+", answer.lower()))
        relevance = len(q_words & a_words) / max(1, len(q_words))
        relevance = min(1.0, relevance)
        return {
            "faithfulness": round(faithfulness, 3),
            "relevance": round(relevance, 3),
            "rationale": "heuristic fallback (no LLM key configured): lexical overlap proxy",
            "mode": "heuristic_fallback",
        }

    prompt = JUDGE_PROMPT_TEMPLATE.format(context=context, question=question, answer=answer)
    try:
        if provider == "anthropic":
            raw, _, _ = _call_anthropic(prompt)
        else:
            raw, _, _ = _call_openai(prompt)
        parsed = _parse_judge_json(raw)
        parsed["mode"] = f"llm_judge:{provider}"
        return parsed
    except Exception as e:
        # Robust handling of malformed JSON / API failure -> fall back rather than crash the run.
        faithfulness = groundedness_heuristic(answer, context_chunks)
        return {
            "faithfulness": round(faithfulness, 3),
            "relevance": 0.5,
            "rationale": f"judge call/parse failed ({e}); used heuristic fallback",
            "mode": "heuristic_fallback_after_error",
        }
