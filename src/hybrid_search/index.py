"""An in-memory index with two retrievers: lexical (BM25) and vector."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from .embeddings import HashingEmbedder, tokens


@dataclass
class Index:
    embedder: object = field(default_factory=HashingEmbedder)
    _docs: Dict[str, str] = field(default_factory=dict)
    _vecs: Dict[str, np.ndarray] = field(default_factory=dict)
    _toks: Dict[str, List[str]] = field(default_factory=dict)

    def add(self, doc_id: str, text: str) -> None:
        self._docs[doc_id] = text
        self._vecs[doc_id] = self.embedder.embed(text)
        self._toks[doc_id] = tokens(text)

    def text(self, doc_id: str) -> str:
        return self._docs.get(doc_id, "")

    def keyword_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Okapi BM25: the standard lexical ranker (TF saturation + IDF +
        document-length normalization), via ``rank-bm25``."""
        q = tokens(query)
        if not self._toks or not q:
            return []
        doc_ids = list(self._toks)
        bm25 = BM25Okapi([self._toks[d] for d in doc_ids])
        scored = [(d, float(s)) for d, s in zip(doc_ids, bm25.get_scores(q)) if s > 0]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    def vector_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Cosine similarity between the query and each document embedding."""
        q = self.embedder.embed(query)
        scored = [(d, float(np.dot(q, v))) for d, v in self._vecs.items()]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]
