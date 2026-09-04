# src/sqm_ai/triage/__init__.py
#
# CONFLICT RESOLVED: Chapter 2.4 creates `src/sqm_ai/triage.py`
# as a module stub; Chapter 16 creates `src/sqm_ai/triage/` as
# a package. The later definition wins, so the package is what
# ships — and Chapter 2's stub survives here, unchanged, because
# tests/test_triage.py still imports `sqm_ai.triage.classify`
# and monkeypatches `sqm_ai.triage.call_model`.
#
# Chapter 2.4 — src/sqm_ai/triage.py (stub; Chapter 16 fills
# in call_model)
from dataclasses import dataclass

CATEGORIES = ("cosmetic", "dimensional", "material", "functional")


@dataclass(frozen=True)
class Classification:
    category: str
    severity: int
    confidence: float


def call_model(text: str) -> dict:
    raise NotImplementedError("built in Chapter 16")


def classify(text: str) -> Classification:
    if not text.strip():
        raise ValueError("empty NCR description")
    result = Classification(**call_model(text))
    if result.category not in CATEGORIES:
        raise ValueError(f"unknown category: {result.category}")
    if not 0.0 <= result.confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    return result
