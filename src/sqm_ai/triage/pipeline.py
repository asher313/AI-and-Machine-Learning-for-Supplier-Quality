"""Executable local composition; persistence and external routing are separate."""

import uuid

import anthropic
from pydantic import ValidationError

from sqm_ai.errors import SchemaError, SQMError
from sqm_ai.triage.classify import classify
from sqm_ai.triage.guardrails import validate
from sqm_ai.triage.rules import apply_rules, decided
from sqm_ai.triage.schema import (
    NCRInput,
    TriageDecision,
    TriageResult,
)


def triage_one(
    ncr,
    *,
    part_map=None,
    neighbours=None,
    classifier=None,
    owner_map=None,
    allow_auto=False,
):
    """Return an auditable decision, including abstention after expected failures.

    auto means an internal work-queue assignment, never a disposition.
    Missing approved ownership, uncertainty, conflicts, and high severity review.
    """
    ncr = NCRInput.model_validate(ncr)
    if not ncr.defect_description.strip():
        raise ValueError("empty NCR description")
    output = TriageDecision(
        decision_id=str(uuid.uuid4()), ncr_id=ncr.ncr_id
    )
    hits = apply_rules(
        ncr.defect_description, ncr.part_number, part_map or {}
    )
    output.rules_fired = [h.rule for h in hits]
    fields = decided(hits)
    traces = []
    try:
        result, model = (classifier or classify)(
            ncr, neighbours or [], traces=traces
        )
        output.model = model
        output.stage = "rules+model" if hits else "model"
        conflicts = [
            k
            for k, v in fields.items()
            if getattr(result, k) != v
        ]
        merged = result.model_dump()
        # These teaching rules provide conservative proposals; a conflict reviews.
        merged.update(fields)
        result = TriageResult.model_validate(merged)
        result = validate(
            result, ncr.defect_description, ncr.supplier_shortlist
        )
        output.result = result
        owner = (owner_map or {}).get(
            (result.category, result.supplier_id)
        )
        output.car_owner_email = owner
        reasons = []
        if conflicts:
            reasons.append("rule_model_conflict")
        if result.severity >= 4:
            reasons.append("high_severity")
        if result.confidence < 0.90:
            reasons.append("low_confidence")
        if not owner:
            reasons.append("owner_lookup_missing")
        if not allow_auto:
            reasons.append("shadow_mode")
        output.routed = "review" if reasons else "auto"
        output.review_notes = ",".join(reasons) or None
    except (SQMError, ValidationError, anthropic.APIError) as exc:
        # Do not persist rejected reasoning, raw error text, or invented defaults.
        output.review_notes = (
            f"{type(exc).__name__}:{exc.field}"
            if isinstance(exc, SchemaError)
            else type(exc).__name__
        )
    finally:
        output.trace_ids = [t.trace_id for t in traces]
    return output
