# src/sqm_ai/car/measure.py
from rapidfuzz.distance import Levenshtein


def edit_fraction(draft: str, final: str) -> float:
    """0.0 = accepted verbatim, 1.0 = fully rewritten."""
    a, b = " ".join(draft.split()), " ".join(final.split())
    if not a and not b:
        return 0.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))
