"""Swappable embeddings for the vector half of the search.

The real path uses proper semantic embeddings: ``sentence-transformers`` (local
model) or the OpenAI API. A dependency-free hashing embedder is kept for offline,
deterministic runs so tests and CI need no model download and no key, but it is
not semantically smart; pick it only for the mechanics.
"""

from __future__ import annotations

import hashlib
import os
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


class OpenAIEmbedder:
    """Real embeddings from the OpenAI API (the ``openai`` extra).

    Reads ``OPENAI_API_KEY`` from the environment or a local ``.env`` file.
    Vectors are L2-normalized so cosine similarity is a plain dot product.
    """

    def __init__(self, model: str = None) -> None:
        try:
            from dotenv import find_dotenv, load_dotenv

            load_dotenv(find_dotenv(usecwd=True))
        except Exception:
            pass
        from openai import OpenAI

        self._client = OpenAI()
        self.model = model or os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        self.dim = int(os.getenv("OPENAI_EMBED_DIM", "1536"))

    def embed(self, text: str) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=[text], dimensions=self.dim)
        v = np.asarray(resp.data[0].embedding, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else v


def get_embedder(prefer: str = "auto"):
    """Return an embedder by name.

    - ``"openai"``: the OpenAI API (needs ``OPENAI_API_KEY``; see ``.env.example``).
    - ``"sentence-transformers"``: a local model (the ``semantic`` extra).
    - ``"hashing"``: the dependency-free offline embedder.
    - ``"auto"`` (default): sentence-transformers if installed, else hashing. It
      never reaches for OpenAI on its own, so it cannot surprise you with a bill.
    """
    if prefer == "openai":
        return OpenAIEmbedder()
    if prefer == "hashing":
        return HashingEmbedder()
    if prefer in ("auto", "sentence-transformers"):
        try:
            return SentenceTransformerEmbedder()
        except Exception:
            if prefer == "sentence-transformers":
                raise
    return HashingEmbedder()
