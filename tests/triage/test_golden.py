# tests/triage/test_golden.py  (marked: nightly)
import json
import pytest
from sqm_ai.triage.pipeline import triage_one

pytestmark = pytest.mark.nightly
GOLDEN = json.load(open("tests/data/golden_ncrs.json"))


def test_category_agreement_above_threshold():
    hits = sum(triage_one(g["ncr"]).category == g["category"]
               for g in GOLDEN)
    assert hits / len(GOLDEN) >= 0.92


def test_no_missed_severity_five():
    """Under-calling a safety-critical NCR is the one
    error that is never acceptable at any rate."""
    assert [] == [
        g for g in GOLDEN if g["severity"] == 5
        and triage_one(g["ncr"]).severity < 5]
