"""Explicit opt-in evaluation; no corpus is loaded at collection time."""

import json
import os
from pathlib import Path

import pytest

from sqm_ai.triage.pipeline import triage_one

pytestmark = [pytest.mark.llm, pytest.mark.nightly]


@pytest.fixture(scope="module")
def evaluated():
    path = os.environ.get("SQM_GOLDEN_NCRS")
    if not path:
        pytest.skip(
            "set SQM_GOLDEN_NCRS to an approved, independently labelled evaluation corpus"
        )
    golden = json.loads(Path(path).read_text())
    if not golden:
        pytest.fail("empty evaluation corpus")
    return [(g, triage_one(g["ncr"])) for g in golden]


def test_category_agreement_above_threshold(evaluated):
    hits = sum(
        d.result is not None
        and d.result.category == g["category"]
        for g, d in evaluated
    )
    assert hits / len(evaluated) >= 0.92


def test_no_missed_severity_five(evaluated):
    safety = [(g, d) for g, d in evaluated if g["severity"] == 5]
    assert safety, "evaluation needs severity-five cases"
    assert not [
        g
        for g, d in safety
        if d.result is None or d.result.severity < 5
    ]
