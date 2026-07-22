"""
Local, dependency-free embedding function.

Why this exists: ChromaDB's DefaultEmbeddingFunction downloads an ONNX
MiniLM model from a host outside this sandbox's egress allowlist, and a
full sentence-transformers/torch install was too large for available disk.
This is a SANDBOX WORKAROUND, not the recommended production choice.

Implementation: TF-IDF-weighted feature hashing (scikit-learn's
HashingVectorizer + TfidfTransformer), L2-normalized -> cosine similarity
over hashed n-gram features. It captures lexical/keyword overlap well
(adequate for this factual corpus + eval set) but does NOT capture deep
semantic similarity the way a transformer embedding would (e.g. synonyms
with no lexical overlap will retrieve poorly). This limitation, and the
recommended swap-in (sentence-transformers all-MiniLM-L6-v2, 384-dim,
or an API embedding model), is documented in the README.
"""
from typing import List
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
import numpy as np


class LocalHashingEmbeddingFunction:
    def __init__(self, dim: int = 384):
        self.dim = dim
        self._pipeline = Pipeline([
            ("hash", HashingVectorizer(n_features=dim, alternate_sign=False, ngram_range=(1, 2))),
            ("tfidf", TfidfTransformer()),
        ])
        # Fit TfidfTransformer's idf on a tiny bootstrap corpus; HashingVectorizer
        # itself needs no fitting (stateless hashing trick).
        self._pipeline.fit(["bootstrap document to initialize idf weights"])

    def __call__(self, input: List[str]) -> List[List[float]]:
        vecs = self._pipeline.transform(input).toarray()
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return vecs.tolist()

    # Chroma's newer client calls embed_documents/embed_query directly
    # rather than always going through __call__.
    def embed_documents(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self(input)

    def name(self) -> str:
        return "local-hashing-tfidf-384"
