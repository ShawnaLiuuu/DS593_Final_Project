"""
retriever.py
Three retrieval modes: BM25, semantic (cosine), hybrid (weighted combination).
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from chunker import Chunk
from embedder import Embedder


class Retriever:
    def __init__(self, chunks: list[Chunk], method: str = "semantic",
                 embedder: Embedder = None, hybrid_alpha: float = 0.5):
        """
        method: "bm25", "semantic", or "hybrid"
        hybrid_alpha: weight for semantic score (1-alpha = BM25 weight)
        """
        self.chunks = chunks
        self.method = method
        self.hybrid_alpha = hybrid_alpha
        self.texts = [c.text for c in chunks]

        # BM25 index
        tokenized = [t.lower().split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)

        # Semantic index
        if method in ("semantic", "hybrid"):
            assert embedder is not None, "Embedder required for semantic/hybrid retrieval"
            self.embedder = embedder
            print("Building semantic index...")
            self.embeddings = embedder.embed(self.texts)  # (N, D)
        else:
            self.embedder = None
            self.embeddings = None

    def _bm25_scores(self, query: str) -> np.ndarray:
        return np.array(self.bm25.get_scores(query.lower().split()))

    def _semantic_scores(self, query: str) -> np.ndarray:
        q_emb = self.embedder.embed_query(query)
        return self.embeddings @ q_emb  # cosine sim (already normalized)

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        mn, mx = scores.min(), scores.max()
        if mx == mn:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    def retrieve(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.method == "bm25":
            scores = self._bm25_scores(query)
        elif self.method == "semantic":
            scores = self._semantic_scores(query)
        elif self.method == "hybrid":
            bm25_scores = self._normalize(self._bm25_scores(query))
            sem_scores = self._normalize(self._semantic_scores(query))
            scores = self.hybrid_alpha * sem_scores + (1 - self.hybrid_alpha) * bm25_scores
        else:
            raise ValueError(f"Unknown retrieval method: {self.method}")

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]
