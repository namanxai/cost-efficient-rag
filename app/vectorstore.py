"""
ChromaDB wrapper: embedded (no server/pods), on-disk persistence,
built-in ONNX MiniLM embedding function (no torch dependency),
metadata filtering support.

Why ChromaDB over pgvector/Qdrant/etc for this exercise:
- Zero infra to stand up (no Postgres, no docker daemon requirement) -> fastest
  path to a genuinely runnable deliverable.
- Embedded mode still gives real cosine-similarity ANN search + metadata filters,
  which is what the eval harness needs to be honest.
- Persists to a local directory, so cost model = disk, not always-on compute pods,
  which is exactly the contrast the assignment wants against a managed vector DB.
"""
import time
import chromadb
from typing import List, Dict, Optional

from app.config import settings
from app.local_embeddings import LocalHashingEmbeddingFunction


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        # NOTE: Chroma's DefaultEmbeddingFunction downloads an ONNX MiniLM model
        # from a host that isn't reachable from this sandbox's restricted egress
        # allowlist, so we use a local, dependency-free embedding function instead.
        # In a normal environment with network/API access, swap this for
        # sentence-transformers("all-MiniLM-L6-v2") or an API embedding model --
        # see README "Embedding model" section.
        self.embedding_fn = LocalHashingEmbeddingFunction(dim=settings.EMBEDDING_DIM)
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: List[Dict]) -> int:
        """
        Upsert by deterministic id -> idempotent re-ingest.
        Returns number of chunks upserted (new or updated).
        """
        if not chunks:
            return 0
        self.collection.upsert(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self.collection.count()

    def count_chunks_for_sources(self, sources: List[str]) -> int:
        """Total number of stored chunks whose source is in `sources`.
        Used by the eval harness to correctly normalize nDCG when relevance
        judgments are doc-level but a doc may contribute multiple chunks."""
        if not sources:
            return 0
        res = self.collection.get(where={"source": {"$in": sources}})
        return len(res["ids"])

    def query(self, query_text: str, top_k: int, source_filter: Optional[str] = None):
        where = {"source": source_filter} if source_filter else None
        t0 = time.perf_counter()
        res = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where,
        )
        latency_ms = (time.perf_counter() - t0) * 1000

        hits = []
        ids = res["ids"][0]
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res["distances"][0]  # cosine distance -> similarity = 1 - distance
        for i in range(len(ids)):
            hits.append({
                "id": ids[i],
                "text": docs[i],
                "metadata": metas[i],
                "similarity": 1 - dists[i],
            })
        return hits, latency_ms


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
