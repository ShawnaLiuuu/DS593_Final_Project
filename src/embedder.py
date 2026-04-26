"""
embedder.py
Wraps sentence-transformers to embed chunks and queries.
Caches embeddings to disk to avoid recomputing on every run.
"""

import numpy as np
import hashlib
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "data/processed"):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)

    def _cache_path(self, texts: list[str]) -> Path:
        key = hashlib.md5((self.model_name + str(len(texts)) + texts[0] + texts[-1]).encode()).hexdigest()
        return self.cache_dir / f"embeddings_{key}.npy"

    def embed(self, texts: list[str]) -> np.ndarray:
        cache_path = self._cache_path(texts)
        if cache_path.exists():
            print(f"Loading cached embeddings from {cache_path}")
            return np.load(cache_path)
        print("Building embeddings (this will be cached for next run)...")
        embeddings = self.model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        np.save(cache_path, embeddings)
        print(f"Embeddings cached to {cache_path}")
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode([query], normalize_embeddings=True)[0]
