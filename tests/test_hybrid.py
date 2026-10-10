from hybrid_search import HashingEmbedder, HybridSearch, reciprocal_rank_fusion

DOCS = {
    "d1": "Returns are free within 30 days.",
    "d2": "You can send items back at no extra cost.",
    "d3": "Free returns and easy refund, explained step by step.",
    "d4": "Our refund policy covers defective products.",
    "d5": "Shipping is fast and fully tracked.",
}


def test_rrf_rewards_agreement():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]]))
    assert fused["b"] > fused["c"]          # b ranked by both beats c ranked by one
    assert fused["a"] > fused["d"]


def test_rrf_includes_items_from_any_list():
    fused = dict(reciprocal_rank_fusion([["a"], ["z"]]))
    assert "a" in fused and "z" in fused


def _hs():
    hs = HybridSearch(embedder=HashingEmbedder())
    hs.add_many(DOCS)
    return hs


def test_keyword_bm25_ranks_by_term_match():
    kws = dict(_hs().index.keyword_search("refund"))
    assert "d4" in kws and "d3" in kws   # both mention "refund"
    assert "d5" not in kws               # shipping doc shares no terms


def test_hybrid_top_found_by_both():
    hits = _hs().search("free returns refund", k=3)
    assert hits[0].doc_id == "d3"
    assert set(hits[0].found_by) == {"keyword", "vector"}


def test_hybrid_surfaces_a_vector_only_doc():
    hits = {h.doc_id: h for h in _hs().search("free returns refund", k=5)}
    # d2 shares no query terms, so keyword misses it; it still appears via vector.
    assert "d2" in hits and "keyword" not in hits["d2"].found_by
