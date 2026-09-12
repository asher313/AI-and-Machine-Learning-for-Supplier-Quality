from importlib import import_module
from types import SimpleNamespace as NS

import pytest

from sqm_ai.errors import SchemaError
from sqm_ai.llm import MODELS
from sqm_ai.triage.guardrails import validate
from sqm_ai.triage.pipeline import triage_one
from sqm_ai.triage.schema import NCRInput, TriageResult

C = import_module("sqm_ai.triage.classify")
VALID = {
    "category": "dimensional",
    "severity": 3,
    "supplier_id": "S-0417",
    "suggested_disposition": "rework",
    "confidence": 0.92,
    "reasoning": "Hole position is outside drawing tolerance on a mounting feature.",
}
SAMPLE = {
    "ncr_id": "NCR-2026-0042",
    "part_number": "7741-B",
    "quantity": 12,
    "defect_description": "Hole 2 mm out of position",
    "supplier_shortlist": ["S-0417"],
    "data_classification": "synthetic",
}


class Block(NS):
    def model_dump(self, **kwargs):
        return vars(self)


def fake_client(payload, calls):
    def create(**kwargs):
        calls.append(kwargs)
        return NS(
            content=[
                Block(
                    type="tool_use",
                    name="record_triage",
                    input=payload,
                )
            ],
            usage=NS(
                input_tokens=1000,
                output_tokens=96,
                cache_read_input_tokens=0,
                cache_creation_input_tokens=0,
            ),
            model=kwargs["model"],
            stop_reason="tool_use",
        )

    return NS(messages=NS(create=create))


def test_tool_block_is_parsed_and_haiku_has_no_effort(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(C, "client", fake_client(VALID, calls))
    traces = []
    result, _ = C.call_model(
        MODELS["fast"], NCRInput(**SAMPLE), [], traces=traces
    )
    assert (
        result.category == "dimensional"
        and "output_config" not in calls[0]
    )
    assert len(traces) == 1 and traces[0].cost_usd > 0
    assert len(traces[0].request_hash) == 64


def test_fabricated_supplier_is_rejected():
    with pytest.raises(SchemaError):
        validate(
            TriageResult(**(VALID | {"supplier_id": "S-9999"})),
            "hole out of tol",
            ["S-0417"],
        )


def test_pipeline_shadow_auto_and_rule_conflict():
    def stub(ncr, nbrs, **kwargs):
        return TriageResult(**VALID), "offline-stub"

    owner = {("dimensional", "S-0417"): "ravi@example.invalid"}
    assert (
        triage_one(
            SAMPLE, classifier=stub, owner_map=owner
        ).routed
        == "review"
    )
    assert (
        triage_one(
            SAMPLE,
            classifier=stub,
            owner_map=owner,
            allow_auto=True,
        ).routed
        == "auto"
    )
    safety = triage_one(
        SAMPLE
        | {
            "defect_description": "Cracked bracket with a scratch"
        },
        classifier=stub,
        owner_map=owner,
        allow_auto=True,
    )
    assert (
        safety.result.severity == 5 and safety.routed == "review"
    )
    assert "rule_model_conflict" in safety.review_notes


def test_failure_has_no_invented_result():
    def fail(*args, **kwargs):
        raise SchemaError("supplier_id", "not on shortlist")

    result = triage_one(SAMPLE, classifier=fail)
    assert result.result is None and result.routed == "review"
    assert result.review_notes == "SchemaError:supplier_id"


def test_restricted_input_does_not_reach_direct_client(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(C, "client", fake_client(VALID, calls))
    result = triage_one(
        SAMPLE | {"data_classification": "restricted"}
    )
    assert (
        result.routed == "review"
        and result.result is None
        and calls == []
    )
