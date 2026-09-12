"""Run the offline synthetic triage walkthrough; no network or database."""

import argparse
import json
from pathlib import Path

from sqm_ai.triage.pipeline import triage_one
from sqm_ai.triage.schema import TriageResult


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/triage/replay_cases.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/build3/decisions.jsonl"),
    )
    args = parser.parse_args()
    bundle = json.loads(args.data.read_text())
    if bundle.get("synthetic") is not True or not bundle.get(
        "cases"
    ):
        raise ValueError(
            "nonempty synthetic replay fixture required"
        )
    decisions = []
    for case in bundle["cases"]:

        def replay(ncr, neighbours, *, traces, case=case):
            return TriageResult.model_validate(
                case["scripted_response"]
            ), "offline-scripted-response"

        decision = triage_one(
            case["ncr"],
            part_map={"7741": "S-0417"},
            classifier=replay,
        )
        decisions.append(decision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for decision in decisions:
            stream.write(decision.model_dump_json() + "\n")
    print(
        json.dumps(
            {
                "mode": "offline replay",
                "records": len(decisions),
                "review": sum(
                    d.routed == "review" for d in decisions
                ),
                "model_quality_evaluated": False,
                "api_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
