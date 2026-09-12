"""Generate the shared fictional four-framework protocol replay."""

import argparse
import json
from pathlib import Path


def generate():
    triage = {
        "category": "dimensional",
        "severity": 3,
        "supplier_id": "S-0417",
        "suggested_disposition": "rework",
        "confidence": 0.75,
        "reasoning": "Synthetic teaching judgment: the hole-position discrepancy requires engineering review; no acceptance authority is delegated.",
    }
    calls = [
        ("record_triage", triage),
        ("get_supplier_history", {"supplier_id": "S-0417"}),
        (
            "draft_car",
            {
                "ncr_id": "NCR-2026-0042",
                "supplier_id": "S-0417",
                "root_cause_hypothesis": "Fixture hypothesis: locating setup may have contributed; verify before concluding.",
                "proposed_containment": "Propose segregating the twelve units pending authorized engineering review.",
            },
        ),
    ]
    responses = []
    for i, (name, args) in enumerate(calls):
        responses.append(
            {
                "id": f"msg_synthetic_{i}",
                "type": "message",
                "role": "assistant",
                "model": "synthetic-script",
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "content": [
                    {
                        "type": "tool_use",
                        "id": f"tool_{i}",
                        "name": name,
                        "input": args,
                    }
                ],
            }
        )
    responses.append(
        {
            "id": "msg_synthetic_final",
            "type": "message",
            "role": "assistant",
            "model": "synthetic-script",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "content": [
                {
                    "type": "text",
                    "text": "Propose rework assessment and a CAR investigation for the out-of-position hole. The authorized engineer decides disposition; root cause remains unverified.",
                }
            ],
        }
    )
    return {
        "synthetic": True,
        "purpose": "Protocol replay, not an empirical framework comparison",
        "ncr": {
            "ncr_id": "NCR-2026-0042",
            "supplier_id": "S-0417",
            "asof": "2026-08-18T10:00:00+00:00",
            "description": "Bracket 7741-B: mounting hole 2 mm out of position, twelve units affected.",
        },
        "history": {
            "total_in_window": 3,
            "recent": [
                {
                    "category": "dimensional",
                    "severity": 2,
                    "discovered_at": d,
                }
                for d in [
                    "2026-08-01",
                    "2026-07-15",
                    "2026-06-10",
                ]
            ],
            "note": "Three-row protocol fixture, not Cobalt's canonical monthly/90-day history.",
        },
        "responses": responses,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/agents/replay.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(generate(), f, indent=2)
    print(f"Wrote fictional agent replay to {args.output}")


if __name__ == "__main__":
    main()
