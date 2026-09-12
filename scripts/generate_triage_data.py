"""Generate fictional pipeline fixtures and scripted responses, not a golden set."""

import argparse
import json
from pathlib import Path


def generate():
    cases = []
    descriptions = {
        "cosmetic": "Appearance blemish recorded; acceptance requires human review.",
        "dimensional": "Hole position is 2 mm outside the stated drawing tolerance.",
        "material": "Wrong alloy reported on the material certificate.",
        "functional": "Assembly did not operate during its prescribed functional test.",
    }
    for category, description in descriptions.items():
        for severity in range(1, 6):
            index = len(cases) + 1
            cases.append(
                {
                    "fixture_id": f"case-{index:03d}",
                    "ncr": {
                        "ncr_id": f"SYN-NCR-{index:04d}",
                        "part_number": "7741-B",
                        "quantity": 12,
                        "defect_description": description
                        + f" Synthetic scenario severity context: level {severity}.",
                        "supplier_shortlist": ["S-0417"],
                        "data_classification": "synthetic",
                    },
                    "scripted_response": {
                        "category": category,
                        "severity": severity,
                        "supplier_id": "S-0417",
                        "suggested_disposition": "rework",
                        "confidence": 0.92
                        if severity % 2
                        else 0.7,
                        "reasoning": "Scripted software fixture; a human must assess acceptance and disposition.",
                    },
                }
            )
    return {
        "synthetic": True,
        "purpose": "Pipeline replay, not model-quality evaluation",
        "cases": cases,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/triage/replay_cases.json"),
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(generate(), stream, indent=2)
        stream.write("\n")
    print(f"Wrote 20 synthetic replay cases to {args.output}")


if __name__ == "__main__":
    main()
