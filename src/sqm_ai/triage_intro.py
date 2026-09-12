"""Chapter 2 testing example, separate from the Chapter 16 package."""

from dataclasses import dataclass

CATEGORIES = ("cosmetic", "dimensional", "material", "functional")


@dataclass(frozen=True)
class Classification:
    category: str
    severity: int
    confidence: float


def call_model(text: str) -> dict:
    from sqm_ai.structured import classify as classify_text

    result = classify_text(text)
    return {
        "category": result.category,
        "severity": result.severity,
        "confidence": result.confidence,
    }


def classify(text: str) -> Classification:
    if not text.strip():
        raise ValueError("empty NCR description")
    result = Classification(**call_model(text))
    if result.category not in CATEGORIES:
        raise ValueError(f"unknown category: {result.category}")
    if not 0.0 <= result.confidence <= 1.0:
        raise ValueError("confidence must be in [0, 1]")
    if not 1 <= result.severity <= 5:
        raise ValueError("severity must be in [1, 5]")
    return result
