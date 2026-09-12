# Chapter 17 retrieval walkthrough

Run from the installed repository root:

```bash
python scripts/generate_retrieval_data.py
python scripts/index_corpus.py
```

This writes 13 fictional source documents and a 25-chunk index into ignored
`data/` and `artifacts/` directories. Both commands protect existing output
files. The default vectors are deterministic lexical hashes, labelled with
their own embedding space. No model weights or API credentials are needed.
These sources contain no licensed AS9100 text and do not reproduce the fictional
121,488-chunk corpus or its reported semantic-quality metrics.

`--embedding bge` on the indexer explicitly loads pinned
`BAAI/bge-large-en-v1.5`, using a query instruction, normalized 1,024-dimensional
vectors, and input-length checks. It requires model weights and enough memory.
The optional reranker also loads lazily at a pinned revision. No model download
occurs merely from importing these modules. Stored token counts in this demo
are labelled UTF-8-byte estimates, not provider token counts. Recount the full
request with the selected generation model before sending it.

For PostgreSQL, install pgvector for the exact PostgreSQL server version, enable
it in a new teaching database, and apply `sql/doc_chunks.sql` in schema `sqm`.
Use the indexer's explicit `--database-url` only for that reviewed destination.
It inserts rather than replacing existing rows. A duplicate document revision,
chunk index, and embedding space fails. Real updates require a versioning
transaction and governed source/access metadata.

Dense search, BM25 candidate construction, and final ID-to-content loading all
apply the same source, program, validity, and embedding-space predicates.
Missing visibility metadata is not shared access. The BM25 implementation
rebuilds over eligible rows per query; this is an inspectable teaching baseline,
not a large-scale sparse-index architecture. Parent expansion checks the
parent's access independently and binds it to the same document revision.

The SQL uses transaction-local `set_config` and pgvector 0.8+ iterative scans.
A WHERE clause enforces returned-row eligibility but does not guarantee that
an approximate index physically filters before generating candidates. Measure
ANN recall against exact filtered search on representative data.

Validation: four offline embedding/chunk/rank/rerank contract tests, two isolated
PostgreSQL 16 + pgvector 0.8.6 integration tests covering dense/sparse/final/parent
access paths, and the generated index walkthrough. Neural retrieval quality,
large-corpus performance, and production identity integration were not tested.

Primary references checked September 11, 2026:

- [pgvector](https://github.com/pgvector/pgvector)
- [pgvector Python adapters](https://github.com/pgvector/pgvector-python)
- [BGE embedding model card](https://huggingface.co/BAAI/bge-large-en-v1.5)
- [E5 model card](https://huggingface.co/intfloat/e5-large-v2)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
