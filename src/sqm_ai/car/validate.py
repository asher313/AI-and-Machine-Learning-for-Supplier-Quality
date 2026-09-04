# src/sqm_ai/car/validate.py
from sqm_ai.car.state import CarState

REQUIRED = {
    "problem": ["problem_statement", "impact"],
    "root_cause": ["root_causes"],
    "actions": ["containment", "corrective_actions"],
    "verify": ["verification_plan"],
}


def validate_node(state: CarState) -> dict:
    """Structural checks. Never judges whether text is good."""
    out = []
    for node, fields in REQUIRED.items():
        for f in fields:
            if not state.get(f):
                out.append(f"FAIL:{node}:{f} is empty")

    causes = state.get("root_causes") or []
    if len(causes) < 3:
        out.append("FAIL:root_cause:fewer than 3 hypotheses")
    for c in causes:
        if len(c.whys) != 5:
            out.append("FAIL:root_cause:not five whys")
        if any(not w.evidence_ref for w in c.whys):
            out.append("FAIL:root_cause:a why cites nothing")

    plan = state.get("verification_plan", "")
    for word in ("sample", "days", "metric"):
        if word not in plan.lower():
            out.append(f"FAIL:verify:no {word} in plan")

    bumped = dict(state.get("retries", {}))
    for w in out:
        node = w.split(":")[1]
        bumped[node] = bumped.get(node, 0) + 1
    # failures replaces; warnings accumulates.
    return {"failures": out, "warnings": out, "retries": bumped}
