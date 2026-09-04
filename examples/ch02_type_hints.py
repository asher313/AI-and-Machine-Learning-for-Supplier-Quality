# Chapter 2 — Type hints
from collections.abc import Sequence


def fpy(passed: Sequence[int], inspected: Sequence[int]) -> float:
    """First-pass yield: total passed / total inspected."""
    total = sum(inspected)
    if total == 0:
        raise ValueError("no units inspected")
    return sum(passed) / total
