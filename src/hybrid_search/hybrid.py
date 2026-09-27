"""Hybrid search: run the lexical and vector retrievers, then fuse with RRF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .embeddings import get_embedder
from .index import Index
from .rrf import reciprocal_rank_fusion


@dataclass
class Hit:
    doc_id: str
    text: str
    score: float
    found_by: List[str]


class HybridSearch:
    def __init__(self, embedder=None) -> None:
        self.index = Index(embedder=embedder or get_embedder())

    def add(self, doc_id: str, text: str) -> None:
        self.index.add(doc_id, text)

    def add_many(self, docs: dict) -> None:
        for doc_id, text in docs.items():
            self.add(doc_id, text)

    def search(self, query: str, k: int = 5, rrf_k: int = 60) -> List[Hit]:
        kw = self.index.keyword_search(query, k=k * 2)
        vec = self.index.vector_search(query, k=k * 2)
        kw_ids = [d for d, _ in kw]
        vec_ids = [d for d, _ in vec]
        fused = reciprocal_rank_fusion([kw_ids, vec_ids], k=rrf_k)
        kw_set, vec_set = set(kw_ids), set(vec_ids)
        hits = []
        for doc_id, score in fused[:k]:
            found_by = []
            if doc_id in kw_set:
                found_by.append("keyword")
            if doc_id in vec_set:
                found_by.append("vector")
            hits.append(Hit(doc_id, self.index.text(doc_id), round(score, 5), found_by))
        return hits
