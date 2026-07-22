"""
Ingestion pipeline: load -> chunk -> embed+store (idempotent via upsert on
deterministic ids).
"""
import os
from typing import Dict

from app.config import settings
from app.loaders import load_document
from app.chunking import build_chunks
from app.vectorstore import get_store


def ingest_file(path: str, chunk_size: int = None, overlap: int = None) -> Dict:
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    text = load_document(path)
    source = os.path.basename(path)
    chunks = build_chunks(source, text, chunk_size, overlap)

    store = get_store()
    n = store.upsert_chunks(chunks)
    return {
        "source": source,
        "chunks_ingested": n,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "total_vectors_in_store": store.count(),
    }


def ingest_directory(dir_path: str = None, chunk_size: int = None, overlap: int = None) -> Dict:
    dir_path = dir_path or settings.DATA_DIR
    results = []
    for fname in sorted(os.listdir(dir_path)):
        fpath = os.path.join(dir_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".pdf", ".html", ".htm", ".md", ".markdown", ".txt")):
            results.append(ingest_file(fpath, chunk_size, overlap))
    return {"files_processed": len(results), "details": results}


if __name__ == "__main__":
    summary = ingest_directory()
    print(summary)
