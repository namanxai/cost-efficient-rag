"""
Word-based sliding-window chunking with configurable size + overlap.
"""
import hashlib
from typing import List, Dict


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split text into overlapping word-windows."""
    words = text.split()
    if not words:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)  # guard against bad config

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks


def chunk_id(source: str, chunk_index: int, chunk_text_value: str) -> str:
    """
    Deterministic ID from source + index + content hash.
    Re-ingesting the same content produces the same ID -> idempotent upsert,
    no duplicate vectors even if ingest is re-run.
    """
    h = hashlib.sha256(chunk_text_value.encode("utf-8")).hexdigest()[:16]
    return f"{source}::{chunk_index}::{h}"


def build_chunks(source: str, text: str, chunk_size: int, overlap: int) -> List[Dict]:
    raw_chunks = chunk_text(text, chunk_size, overlap)
    result = []
    for i, c in enumerate(raw_chunks):
        result.append({
            "id": chunk_id(source, i, c),
            "text": c,
            "metadata": {
                "source": source,
                "chunk_index": i,
                "chunk_size": chunk_size,
                "overlap": overlap,
            },
        })
    return result
