"""Swappable embeddings for the vector half of the search.

Default is a dependency-free hashing embedder (offline, deterministic) so the
fusion mechanics run anywhere. For real semantic matches (paraphrases, synonyms)
use sentence-transformers via ``get_embedder("sentence-transformers")``.
"""

from __future__ import annotations

import hashlib
import re
from typing import List

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("a an and are as at be by for from in is it of on or the to with".split())


def tokens(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class HashingEmbedder:
    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for t in tokens(text):
            v[int(hashlib.md5(t.encode()).hexdigest(), 16) % self.dim] += 1.0
        n = float(np.linalg.norm(v))
        return v / n if n else v


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._m = SentenceTransformer(model_name)
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._m.encode([text], normalize_embeddings=True)[0], dtype=np.float32)


def get_embedder(prefer: str = "auto"):
    if prefer in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if prefer == "sentence-transformers":
                raise
    return HashingEmbedder()
