# src/sqm_ai/triage/pipeline.py
#
# EXAMPLE_CANON lists pipeline.py in src/sqm_ai/triage/, and
# tests/triage/test_golden.py imports triage_one from it, but
# Chapter 16 prints the stages (rules, classify, guardrails,
# trace, review) rather than the composition itself.
#
# TODO(book): condensed in Chapter 16 — the composition is
# described across 16.4-16.8 but never listed. Complete
# before production use.
from sqm_ai.triage.classify import classify_ncr
from sqm_ai.triage.guardrails import validate
from sqm_ai.triage.rules import apply_rules
from sqm_ai.triage.schema import TriageResult


def triage_one(ncr: dict) -> TriageResult:
    """Rules first, model second, guardrails always.

    Section 16.4: the deterministic rules settle roughly a
    third of the traffic without spending a token.
    Section 16.5: the rest go to the model with the taxonomy
    and retrieved neighbours.
    Section 16.6: every answer is validated before it is
    stored or routed.
    """
    raise NotImplementedError(
        "Chapter 16 lists the stages, not the composition"
    )
