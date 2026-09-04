# src/sqm_ai/retrieval/rerank.py
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")


def rerank(
    question: str, shortlist: list[dict], k: int = 8
) -> list[dict]:
    """Reorder a first-stage pool by true relevance."""
    pairs = [(question, c["content"]) for c in shortlist]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(shortlist, scores),
        key=lambda pair: pair[1], reverse=True)
    return [c for c, _ in ranked[:k]]
