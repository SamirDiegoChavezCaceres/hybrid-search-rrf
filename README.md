# hybrid-search-rrf

Combine **keyword** (lexical) and **vector** (semantic) search into one ranking
with **Reciprocal Rank Fusion**, the retrieval setup behind most good RAG.

## Demo

![demo](assets/demo.gif)

The demo (`scripts/demo.py`) runs offline on five short support-FAQ snippets
(returns, refunds, shipping) and the query *"free returns refund"*. It prints the
keyword-only ranking, the vector-only ranking, and the Reciprocal Rank Fusion of
the two side by side, so you can see a document that ranks well in both rise to
the top without any score calibration between the retrievers. Set
`HS_EMBEDDER=sentence-transformers` for real semantic matches.

## Why hybrid

- **Keyword** search nails exact terms: product codes, names, error strings,
  numbers. It has no idea about meaning.
- **Vector** search catches paraphrases and synonyms. It often misses an exact
  rare token because the embedding blurs it.

Neither alone is enough. You run both and fuse the two rankings.

## Why RRF

The two retrievers return incomparable scores (an overlap count vs a cosine), so
you cannot just add them. **Reciprocal Rank Fusion** ignores the scores and uses
only each item's rank:

```
score(doc) = sum over retrievers of  1 / (k + rank)
```

A document ranked well by both rises to the top, and a document found by only one
still makes the list. No score calibration, one tunable constant (`k`, default 60).

## Use it

```python
from hybrid_search import HybridSearch

hs = HybridSearch()
hs.add_many({"d1": "Returns are free within 30 days.", ...})

for hit in hs.search("free returns refund"):
    print(hit.doc_id, hit.found_by, hit.text)   # found_by: ['keyword', 'vector']
```

```bash
pip install -e .
python scripts/demo.py
```

The demo shows a document that keyword search misses entirely still surfacing
through the vector retriever. Real semantic matches (true paraphrases) need real
embeddings:

```bash
pip install -e ".[semantic]"
HS_EMBEDDER=sentence-transformers python scripts/demo.py
```

By default the vector half uses a dependency-free hashing embedder, so the fusion
mechanics run offline; it is not semantically smart. That is the same honest
limitation as a bag-of-words model: swap in sentence-transformers for real
retrieval quality.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

Covers the RRF math (rewards agreement, keeps single-list items), exact keyword
overlap, and that the fused ranking tags each hit with which retriever found it.

## Limitations and next steps

- The hashing embedder shows mechanics only; semantic quality needs
  sentence-transformers or a hosted embedder.
- Keyword search is plain token overlap; BM25 would weight rare terms better.
- Next: per-field boosting, and a reranker (cross-encoder) over the fused top-k.

## License

MIT.
