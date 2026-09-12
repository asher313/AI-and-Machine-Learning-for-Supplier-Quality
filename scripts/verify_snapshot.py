"""Inspect configured model IDs; metadata listing is not a health or retirement test."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from sqm_ai.llm import MODELS, get_client


def inspect_catalog(ids, models=MODELS):
    if not isinstance(ids, list) or any(
        not isinstance(x, str) or not x for x in ids
    ):
        raise ValueError(
            "model IDs must be a list of nonempty strings"
        )
    available = set(ids)
    return [
        {
            "tier": tier,
            "model_id": model,
            "status": "LISTED"
            if model in available
            else "NOT_LISTED",
        }
        for tier, model in models.items()
    ]


def read_live(client):
    # Iterating this SDK page traverses all pages; an error fails the complete check.
    return [m.id for m in client.models.list(limit=100)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--catalog", type=Path)
    mode.add_argument(
        "--live",
        action="store_true",
        help="Explicitly query provider metadata using configured credentials",
    )
    mode.add_argument("--write-fixture", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.write_fixture:
        args.write_fixture.parent.mkdir(
            parents=True, exist_ok=True
        )
        args.write_fixture.write_text(
            json.dumps(
                {
                    "kind": "synthetic-model-catalog",
                    "model_ids": list(MODELS.values()),
                },
                indent=2,
            )
            + "\n"
        )
        return 0
    if args.out is None:
        p.error("--out is required for a report")
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "first-party metadata only"
        if args.live
        else "supplied catalog; not live availability",
    }
    try:
        if args.live:
            ids = read_live(get_client())
        else:
            catalog = json.loads(args.catalog.read_text())
            ids = catalog["model_ids"]
            report["catalog_kind"] = catalog.get(
                "kind", "unspecified"
            )
        report["models"] = inspect_catalog(ids)
        status = (
            0
            if all(
                row["status"] == "LISTED"
                for row in report["models"]
            )
            else 1
        )
    except Exception as exc:
        report.update(
            status="CHECK_FAILED", error_type=type(exc).__name__
        )
        status = 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
