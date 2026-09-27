"""An in-memory index with two retrievers: lexical (keyword) and vector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .embeddings import HashingEmbedder, tokens


@dataclass
class Index:
    embedder: object = field(default_factory=HashingEmbedder)
    _docs: Dict[str, str] = field(default_factory=dict)
    _vecs: Dict[str, np.ndarray] = field(default_factory=dict)
    _toks: Dict[str, set] = field(default_factory=dict)

    def add(self, doc_id: str, text: str) -> None:
        self._docs[doc_id] = text
        self._vecs[doc_id] = self.embedder.embed(text)
        self._toks[doc_id] = set(tokens(text))

    def text(self, doc_id: str) -> str:
        return self._docs.get(doc_id, "")

    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Lexical overlap: how many query terms appear in the document."""
        q = set(tokens(query))
        scored = [(d, len(q & t)) for d, t in self._toks.items()]
        scored = [(d, s) for d, s in scored if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def vector_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Cosine similarity between the query and each document embedding."""
        q = self.embedder.embed(query)
        scored = [(d, float(np.dot(q, v))) for d, v in self._vecs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
