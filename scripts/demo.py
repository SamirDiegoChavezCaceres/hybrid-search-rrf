"""Hybrid search walkthrough: BM25 vs vector vs RRF fusion.

    python scripts/demo.py

Keyword retrieval is always real Okapi BM25. For the vector half the demo uses
OpenAI when OPENAI_API_KEY is set (see .env.example), otherwise an offline
hashing embedder. Force a backend with HS_EMBEDDER=openai|sentence-transformers|hashing.
"""

from __future__ import annotations

import os

from hybrid_search import HybridSearch, get_embedder


def pick_embedder():
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass
    choice = os.getenv("HS_EMBEDDER")
    if not choice:
        choice = "openai" if os.getenv("OPENAI_API_KEY") else "auto"
    try:
        return get_embedder(choice)
    except Exception:
        return get_embedder("hashing")

DOCS = {
    "d1": "Returns are free within 30 days.",
    "d2": "You can send items back at no extra cost.",
    "d3": "Free returns and easy refund, explained step by step.",
    "d4": "Our refund policy covers defective products.",
    "d5": "Shipping is fast and fully tracked.",
}
QUERY = "free returns refund"


def rule(title: str) -> None:
    print(f"\n=== {title} ===")


def show(pairs, index) -> None:
    for doc_id, score in pairs:
        print(f"    {doc_id}  {score:>6}  {index.text(doc_id)}")


def main() -> None:
    embedder = pick_embedder()
    hs = HybridSearch(embedder=embedder)
    hs.add_many(DOCS)
    print(f"query: {QUERY!r}  |  vector embedder: {type(embedder).__name__}")

    rule("keyword only (Okapi BM25)")
    show([(d, round(s, 3)) for d, s in hs.index.keyword_search(QUERY, 3)], hs.index)

    rule("vector only (cosine)")
    show([(d, round(s, 3)) for d, s in hs.index.vector_search(QUERY, 3)], hs.index)

    rule("hybrid (Reciprocal Rank Fusion)")
    for hit in hs.search(QUERY, k=4):
        print(f"    {hit.doc_id}  {hit.score:>7}  [{'+'.join(hit.found_by)}]  {hit.text}")

    rule("why fuse")
    print("  RRF needs no score calibration between the two retrievers; a doc both")
    print("  rank well rises to the top. Keyword catches exact terms, vector catches")
    print("  wording the keyword search misses (strongest with real embeddings).")


if __name__ == "__main__":
    main()
