"""Deterministic shape/reference checks; source entailment still requires review."""

from pydantic import ValidationError

from sqm_ai.car.nodes import SCHEMAS

FIELDS = {
    "problem": "problem",
    "root_cause": "root_cause",
    "actions": "actions",
    "verify": "verification_plan",
}


def validate_node(state):
    failures, gaps = (
        [],
        list(state["evidence"].get("investigation_gaps", [])),
    )
    ids = {s["id"] for s in state["evidence"]["sources"]}
    for role, field in FIELDS.items():
        try:
            value = (
                SCHEMAS[role]
                .model_validate(state.get(field, {}))
                .model_dump()
            )
        except ValidationError:
            failures.append(
                {
                    "owner": role,
                    "reason": "missing or invalid structured section",
                }
            )
            continue
        refs = value.get("evidence_refs", [])
        gaps.extend(value.get("investigation_gaps", []))
        if role == "root_cause":
            candidates = [
                h["candidate_cause"].strip().casefold()
                for h in value["hypotheses"]
            ]
            if len(candidates) != len(set(candidates)):
                failures.append(
                    {
                        "owner": role,
                        "reason": "duplicate candidate causes",
                    }
                )
            for h in value["hypotheses"]:
                gaps.extend(h["investigation_gaps"])
                for why in h["whys"]:
                    refs += why["evidence_refs"]
                    if why["investigation_gap"]:
                        gaps.append(why["investigation_gap"])
        if any(ref not in ids for ref in refs):
            failures.append(
                {
                    "owner": role,
                    "reason": "reference is absent from authorized evidence",
                }
            )
    return {
        "failures": failures,
        "gaps": list(dict.fromkeys(gaps)),
        "retry_target": failures[0]["owner"]
        if failures
        else None,
        "warnings": [
            f"{f['owner']}: {f['reason']}" for f in failures
        ],
    }
