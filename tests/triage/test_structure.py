# tests/triage/test_structure.py
import pytest
from pydantic import ValidationError

from sqm_ai.triage.schema import TriageResult

VALID = {
    "category": "dimensional",
    "severity": 3,
    "supplier_id": "S-0417",
    "suggested_disposition": "rework",
    "confidence": 0.92,
    "reasoning": "Hole position is outside drawing "
    "tolerance on a mounting feature.",
}


def test_valid_result_parses():
    r = TriageResult(**VALID)
    assert 1 <= r.severity <= 5
    assert 0.0 <= r.confidence <= 1.0


@pytest.mark.parametrize(
    "field,bad",
    [
        ("severity", 7),
        ("category", "Dimensional"),
        ("supplier_id", "Cobalt Machining"),
        ("confidence", 1.4),
        ("reasoning", "too short"),
    ],
)
def test_bad_field_rejected(field, bad):
    with pytest.raises(ValidationError):
        TriageResult(**(dict(VALID) | {field: bad}))
