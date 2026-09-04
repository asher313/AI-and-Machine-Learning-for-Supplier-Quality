# scripts/index_corpus.py  (condensed: imports, and the
# TODO(book): condensed in Chapter 17 — complete before production use.
# four loaders behind load_all(), which differ only in how
# they read their source files. recursive_chunks() wraps
# the splitter of 17.3 the way clause_chunks() is written
# there, returning the same dicts.)
rows = []
for doc in load_all():
    chunks = (clause_chunks(doc.text, doc.id)
              if doc.structured
              else recursive_chunks(doc.text, doc.id))
    vecs = embedder.encode(
        [c["content"] for c in chunks],
        normalize_embeddings=True, batch_size=64)
    for chunk, vec in zip(chunks, vecs):
        chunk |= {
            "embedding": vec,
            "embed_model": EMBED_MODEL,
            "token_count": count_tokens(chunk["content"]),
            "metadata": Jsonb(doc.metadata),
            "valid_from": doc.effective_date,
        }
        rows.append(chunk)

with connect() as conn:
    insert_chunks(conn, rows)
