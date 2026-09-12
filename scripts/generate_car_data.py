"""Generate fictional CAR role outputs and review fixtures; no model-quality labels."""

import argparse
import json
from pathlib import Path


def generate():
    evidence = {
        "ncr_id": "NCR-2026-0042",
        "supplier_id": "S-0417",
        "investigation_gaps": [],
        "sources": [
            {
                "id": "NCR-2026-0042",
                "text": "Fictional NCR: Cobalt Machining bracket 7741-B has a mounting hole 2 mm out of position; twelve units are affected. Engineering disposition is pending.",
            },
            {
                "id": "SYN-INSPECTION-1",
                "text": "Fictional inspection note: the locating pad has visible wear. Its causal role has not been established.",
            },
            {
                "id": "SYN-PLAN-1",
                "text": "Software fixture only: propose checking 50 units over 30 days against the applicable drawing. This arbitrary plan is not a statistically justified effectiveness demonstration.",
            },
        ],
    }
    roles = {
        "problem": {
            "problem_statement": "Twelve supplied brackets have a mounting hole reported 2 mm out of position.",
            "impact": "Engineering review is needed before any disposition; downstream consequences are not established.",
            "evidence_refs": ["NCR-2026-0042"],
        },
        "root_cause": {
            "hypotheses": [
                {
                    "observation": "A locating pad shows visible wear.",
                    "candidate_cause": "Locating-pad wear may have contributed to position error; this remains unconfirmed.",
                    "confidence": 0.4,
                    "investigation_gaps": [],
                    "whys": [
                        {
                            "question": "What mechanism could change location?",
                            "answer": "Wear may alter positioning; the cited observation motivates a test rather than proving causality.",
                            "status": "inferred",
                            "evidence_refs": ["SYN-INSPECTION-1"],
                            "investigation_gap": None,
                        }
                    ],
                }
            ],
            "investigation_gaps": [],
        },
        "actions": {
            "containment": "Propose segregating the twelve units for authorized review.",
            "corrective_actions": [
                "If engineering confirms locating-pad wear is causal, propose controlled fixture restoration and validation before use."
            ],
            "evidence_refs": [
                "NCR-2026-0042",
                "SYN-INSPECTION-1",
            ],
            "investigation_gaps": [],
        },
        "verify": {
            "sample_size": 50,
            "duration_days": 30,
            "metric": "Hole-position conformity",
            "acceptance_criterion": "Conformance to the applicable approved drawing; no numerical tolerance is invented.",
            "design_basis": "This fixed synthetic plan tests software fields only; an engineer must design a valid effectiveness study.",
            "evidence_refs": ["SYN-PLAN-1"],
            "investigation_gaps": [],
        },
    }
    return {
        "synthetic": True,
        "purpose": "CAR workflow replay, not evidence of causal correctness",
        "supplier_names": {"S-0417": "Cobalt Machining"},
        "evidence": evidence,
        "roles": roles,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/car/replay.json"),
    )
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(generate(), f, indent=2)
    print(f"Wrote fictional CAR replay to {args.output}")


if __name__ == "__main__":
    main()
