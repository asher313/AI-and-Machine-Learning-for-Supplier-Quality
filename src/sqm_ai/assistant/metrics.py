"""Retrieval metrics on adjudicated answerable questions; abstention is separate."""

import math


def _ranking(retrieved, k=None):
    if k is not None and (not isinstance(k, int) or k < 1):
        raise ValueError("positive integer cutoff required")
    if len(retrieved) != len(set(retrieved)):
        raise ValueError("rankings must contain unique ids")


def recall_at_k(retrieved, relevant, k):
    _ranking(retrieved, k)
    if not relevant:
        raise ValueError(
            "no relevance set: evaluate unanswerability separately"
        )
    return len(set(retrieved[:k]) & set(relevant)) / len(
        set(relevant)
    )


def reciprocal_rank(retrieved, relevant):
    _ranking(retrieved)
    if not relevant:
        raise ValueError(
            "no relevance set: evaluate unanswerability separately"
        )
    return next(
        (
            1 / rank
            for rank, item in enumerate(retrieved, 1)
            if item in relevant
        ),
        0.0,
    )


def ndcg_at_k(retrieved, grades, k):
    """Linear-gain nDCG. Report this gain convention with results."""
    _ranking(retrieved, k)
    if any(
        not math.isfinite(v) or v < 0 for v in grades.values()
    ):
        raise ValueError("grades must be finite and nonnegative")

    def dcg(scores):
        return sum(
            value / math.log2(i + 2)
            for i, value in enumerate(scores)
        )

    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0:
        raise ValueError(
            "no positive relevance grades; score refusal separately"
        )
    return (
        dcg([grades.get(item, 0.0) for item in retrieved[:k]])
        / ideal
    )
