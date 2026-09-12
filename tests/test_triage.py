# tests/test_triage.py
import pytest

from sqm_ai import triage_intro as triage


def fake_model(text: str) -> dict:
    return {"category": "dimensional",
            "severity": 3, "confidence": 0.92}


def test_classify_uses_model_output(monkeypatch):
    monkeypatch.setattr(triage, "call_model", fake_model)
    result = triage.classify("hole 2 mm out of tolerance")
    assert result.category == "dimensional"
    assert 0 <= result.confidence <= 1


def test_classify_rejects_unknown_category(monkeypatch):
    def bad_model(text: str) -> dict:
        return {"category": "paperwork",
                "severity": 2, "confidence": 0.7}
    monkeypatch.setattr(triage, "call_model", bad_model)
    with pytest.raises(ValueError, match="unknown category"):
        triage.classify("cert missing")


def test_classify_rejects_empty_text():
    with pytest.raises(ValueError):
        triage.classify("   ")


@pytest.mark.llm
@pytest.mark.parametrize("text,expected", [
    ("Dent on outer housing", "cosmetic"),
    ("Hole 2 mm out of tolerance", "dimensional"),
    ("Wrong material grade shipped", "material"),
])
def test_golden_examples(text, expected):
    assert triage.classify(text).category == expected
