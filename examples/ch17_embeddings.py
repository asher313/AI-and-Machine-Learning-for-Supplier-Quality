# Chapter 17 — 17.4 Embeddings
# np and SentenceTransformer: imports as in §17.2.
def recall_at_k(
    model_name: str, questions: list[tuple[str, int]],
    corpus: list[str], k: int = 10,
) -> float:
    """Share of questions whose right chunk is in top-k."""
    emb = SentenceTransformer(model_name)
    vecs = emb.encode(corpus, normalize_embeddings=True)
    hits = 0
    for question, correct_id in questions:
        q = emb.encode([question],
                       normalize_embeddings=True)[0]
        top = np.argsort(vecs @ q)[::-1][:k]
        hits += int(correct_id in top)
    return hits / len(questions)
