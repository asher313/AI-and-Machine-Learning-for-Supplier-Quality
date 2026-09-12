"""Fail closed on incomplete or unacceptable Build 1 fold metrics."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def probability(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(
            "metric and floor must be finite numbers in [0,1]"
        )
    return float(value)


def assess(metrics, min_auc, min_recall_sev3):
    floors = {
        "roc_auc": probability(min_auc),
        "recall_sev3": probability(min_recall_sev3),
    }
    if not isinstance(metrics, list) or not metrics:
        raise ValueError(
            "nonempty chronological fold list required"
        )
    folds = [row.get("fold") for row in metrics]
    if any(type(f) is not int for f in folds) or folds != list(
        range(1, len(folds) + 1)
    ):
        raise ValueError(
            "ordered contiguous folds starting at one required"
        )
    # Validate all supplied folds; gate the most recent predefined holdout.
    for row in metrics:
        for name in floors:
            probability(row[name])
        if row.get("screen_threshold") != 0.04:
            raise ValueError(
                "recall must use the predefined 0.04 screening threshold"
            )
        for count in ("n_test", "positive_test"):
            if type(row.get(count)) is not int or row[count] <= 0:
                raise ValueError(
                    "positive holdout row/event counts required"
                )
        if row["positive_test"] >= row["n_test"]:
            raise ValueError("holdout needs both classes")
    last = metrics[-1]
    checks = [
        {
            "metric": name,
            "value": last[name],
            "floor": floor,
            "passed": last[name] >= floor,
        }
        for name, floor in floors.items()
    ]
    return {
        "passed": all(c["passed"] for c in checks),
        "fold": last["fold"],
        "n_test": last["n_test"],
        "positive_test": last["positive_test"],
        "screen_threshold": 0.04,
        "checks": checks,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--min-auc", type=float, required=True)
    p.add_argument("--min-recall-sev3", type=float, required=True)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    raw = a.metrics.read_bytes()
    try:
        report = assess(
            json.loads(raw), a.min_auc, a.min_recall_sev3
        )
    except (
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
    ) as exc:
        report = {"passed": False, "error": str(exc)}
    report["metrics_sha256"] = hashlib.sha256(raw).hexdigest()
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for check in report.get("checks", []):
        mark = "PASS" if check["passed"] else "FAIL"
        print(
            f"{mark} {check['metric']}={check['value']:.3f} floor={check['floor']:.2f}"
        )
    print(
        "quality gate passed"
        if report["passed"]
        else "quality gate failed"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
