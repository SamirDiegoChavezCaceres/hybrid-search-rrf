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


def make_answerer():
    """Return a grounded answer(question, context) via OpenAI, or None offline."""
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None
    client = OpenAI()
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    def answer(question: str, context: str) -> str:
        resp = client.chat.completions.create(
            model=model, temperature=0,
            messages=[
                {"role": "system", "content":
                 "Answer the question using ONLY the passages. If they do not cover "
                 "it, say you don't know. One or two sentences on a single line."},
                {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    return answer


DOCS = {
    "d1": "Returns are free within 30 days.",
    "d2": "You can send items back at no extra cost.",
    "d3": "Free returns and easy refund, explained step by step.",
    "d4": "Our refund policy covers defective products.",
    "d5": "Shipping is fast and fully tracked.",
}
QUERY = "free returns refund"
QUESTION = "If I return an item, do I pay anything, and what does the refund cover?"


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

    rule("1. keyword only (Okapi BM25)")
    show([(d, round(s, 3)) for d, s in hs.index.keyword_search(QUERY, 3)], hs.index)

    rule("2. vector only (cosine)")
    show([(d, round(s, 3)) for d, s in hs.index.vector_search(QUERY, 3)], hs.index)

    rule("3. hybrid (Reciprocal Rank Fusion)")
    fused = hs.search(QUERY, k=4)
    for hit in fused:
        print(f"    {hit.doc_id}  {hit.score:>7}  [{'+'.join(hit.found_by)}]  {hit.text}")
    print("  RRF needs no score calibration: a doc both rank well rises to the top.")

    rule("4. end to end: answer generated from the fused top-k")
    top = fused[:3]
    print(f"  Q: {QUESTION}")
    print(f"  context: {', '.join(h.doc_id for h in top)} (the fused top-3)")
    answer = make_answerer()
    if answer:
        context = "\n".join(h.text for h in top)
        print(f"  answer: {answer(QUESTION, context)}")
    else:
        print("  (set OPENAI_API_KEY to generate the grounded answer from these passages)")


if __name__ == "__main__":
    main()
