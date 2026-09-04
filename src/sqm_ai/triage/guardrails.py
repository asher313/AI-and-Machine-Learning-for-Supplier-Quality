# src/sqm_ai/triage/guardrails.py
import re
import structlog

from sqm_ai.errors import SchemaError
from sqm_ai.triage.schema import TriageResult

log = structlog.get_logger()
I = re.IGNORECASE

DIMENSIONAL = re.compile(
    r"\b(tolerance|out of tol|oversize|undersize|mm|inch"
    r"|diameter|position|true position)\b", I)
MARKINGS = re.compile(
    r"\b(ITAR|CUI|export controlled)\b", I)
PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),        # SSN
    re.compile(r"\b\d{13,16}\b"),                # card
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),  # email
    re.compile(r"\b\(?\d{3}\)?[ .-]\d{3}[ .-]\d{4}\b"),
]


def contains_pii(text: str) -> bool:
    """True if the text holds anything person-identifying."""
    return any(p.search(text) for p in PII_PATTERNS)


def validate(
    result: TriageResult, description: str,
    shortlist: list[str],
) -> TriageResult:
    """Raise SchemaError if the result may not be stored."""
    # Policy: the reasoning is stored and read by people.
    if contains_pii(result.reasoning):
        raise SchemaError("reasoning", "contains PII")
    if MARKINGS.search(result.reasoning):
        raise SchemaError("reasoning", "echoes a marking")
    # Structural: only ids we supplied are allowed.
    if result.supplier_id not in shortlist:
        raise SchemaError("supplier_id", "not on shortlist")
    # Semantic: warn, never raise. These are soft signals.
    if (DIMENSIONAL.search(description)
            and result.category != "dimensional"):
        log.warning("triage_category_smell",
                    said=result.category)
    if result.severity >= 4 and result.confidence >= 0.99:
        log.warning("triage_overconfident", sev=result.severity)
    return result
