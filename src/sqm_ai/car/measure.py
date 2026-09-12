"""Text-change proxy, not an estimate of factual correctness or causal impact."""

from rapidfuzz.distance import Levenshtein


def edit_fraction(draft: str, final: str) -> float:
    """Character edit distance / longer normalized length, in [0,1]."""
    a, b = " ".join(draft.split()), " ".join(final.split())
    return Levenshtein.normalized_distance(a, b)
