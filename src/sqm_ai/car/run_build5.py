"""Run or resume a generated CAR replay, with an explicit synthetic human decision."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from langgraph.types import Command

from sqm_ai.car.graph import build_graph
from sqm_ai.car.nodes import CarServices
from sqm_ai.car.redact import Redactor


def replay_services(path):
    data = json.loads(Path(path).read_text())
    if data.get("synthetic") is not True:
        raise ValueError("synthetic replay bundle required")

    def caller(role, context, **kwargs):
        return deepcopy(data["roles"][role])

    return CarServices(
        data["evidence"]["ncr_id"],
        data["evidence"]["supplier_id"],
        Redactor.build(data["supplier_names"]),
        lambda state: deepcopy(data["evidence"]),
        lambda decision: (
            decision.actor_id == "synthetic-reviewer"
        ),
        data_classification="synthetic",
        role_caller=caller,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input", type=Path, default=Path("data/car/replay.json")
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/build5/checkpoints.sqlite"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/build5/draft.json"),
    )
    p.add_argument("--thread-id", default="synthetic-car-review")
    p.add_argument(
        "--decision", choices=["accept_draft", "reject"]
    )
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    services = replay_services(args.input)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "configurable": {"thread_id": args.thread_id},
        "recursion_limit": 64,
    }
    with build_graph(args.checkpoint, services) as app:
        old = app.get_state(config)
        if args.decision:
            if not old.next:
                raise ValueError("no paused run for this thread")
            request = Command(
                resume={
                    "actor_id": "synthetic-reviewer",
                    "action": args.decision,
                    "note": "Explicit CLI decision in a fictional software replay.",
                    "evidence": None,
                }
            )
        else:
            if old.values:
                raise ValueError(
                    "thread already exists; resume it explicitly or choose a new ID"
                )
            request = {
                "ncr_id": services.ncr_id,
                "supplier_id": services.supplier_id,
            }
        result = app.invoke(request, config)
        paused = bool(app.get_state(config).next)
    output = {
        "mode": "synthetic software replay",
        "api_calls": 0,
        "model_quality_evaluated": False,
        "status": "awaiting_review"
        if paused
        else result["status"],
        "draft": result.get("draft"),
        "failures": result.get("failures", []),
        "gaps": result.get("gaps", []),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(output, f, indent=2)
    print(
        json.dumps(
            {k: v for k, v in output.items() if k != "draft"}
        )
    )


if __name__ == "__main__":
    main()
