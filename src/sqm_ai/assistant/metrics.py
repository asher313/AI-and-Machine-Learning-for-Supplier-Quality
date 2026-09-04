# src/sqm_ai/assistant/metrics.py
import math


def recall_at_k(
    retrieved: list[int], relevant: set[int], k: int
) -> float:
    """Share of relevant chunks that appear in the top k."""
    if not relevant:
        return 0.0
    hits = set(retrieved[:k]) & relevant
    return len(hits) / len(relevant)


def reciprocal_rank(
    retrieved: list[int], relevant: set[int]
) -> float:
    """1/rank of the first relevant chunk; 0 if none."""
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[int], grades: dict[int, float], k: int
) -> float:
    """Rank quality with graded relevance, 0 to 1."""
    def dcg(scores):
        return sum(
            s / math.log2(i + 2) for i, s in enumerate(scores)
        )

    got = [grades.get(c, 0.0) for c in retrieved[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    best = dcg(ideal)
    return dcg(got) / best if best > 0 else 0.0
