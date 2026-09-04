# tests/triage/test_pipeline.py  (condensed: the fake
# TODO(book): condensed in Chapter 16 — complete before production use.
# objects carry only the fields the code under test reads)
from types import SimpleNamespace as NS

import pytest
from sqm_ai.errors import SchemaError
from sqm_ai.triage import classify as C
from sqm_ai.triage.guardrails import validate
from sqm_ai.triage.schema import TriageResult
from tests.triage.test_structure import VALID


def fake_client(payload):
    resp = NS(
        content=[NS(type="tool_use", input=payload)],
        usage=NS(input_tokens=412, output_tokens=96,
                 cache_read_input_tokens=3011),
        model="claude-haiku-4-5", stop_reason="tool_use")
    return NS(messages=NS(create=lambda **kw: resp))


def test_tool_block_is_parsed(monkeypatch, sample_ncr):
    monkeypatch.setattr(C, "client", fake_client(VALID))
    result, _ = C.call_model("m", sample_ncr, [])
    assert result.category == "dimensional"
    assert result.confidence == pytest.approx(0.92)


def test_fabricated_supplier_is_rejected():
    bad = TriageResult(
        **(dict(VALID) | {"supplier_id": "S-9999"}))
    with pytest.raises(SchemaError):
        validate(bad, "hole out of tol", ["S-0417"])
