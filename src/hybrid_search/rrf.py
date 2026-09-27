"""Reciprocal Rank Fusion.

Given several ranked lists of the same items, RRF combines them into one ranking
using only each item's *rank*, not its score:

    score(item) = sum over lists of  1 / (k + rank)      (rank is 1-based)

That is the whole trick: because it ignores the raw scores, it fuses a lexical
ranking and a vector ranking without having to calibrate their incomparable
score scales. Items ranked well by several lists rise to the top.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

DEFAULT_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = DEFAULT_K
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
