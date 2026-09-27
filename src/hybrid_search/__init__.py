"""Hybrid search: lexical + vector retrieval fused with Reciprocal Rank Fusion."""

from .embeddings import HashingEmbedder, get_embedder
from .hybrid import Hit, HybridSearch
from .index import Index
from .rrf import reciprocal_rank_fusion

__all__ = [
    "HybridSearch",
    "Hit",
    "Index",
    "reciprocal_rank_fusion",
    "get_embedder",
    "HashingEmbedder",
]
