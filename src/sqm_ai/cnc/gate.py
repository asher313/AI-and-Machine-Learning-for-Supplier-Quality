"""Frozen educational rejection policy; passing is not production approval."""

import argparse
import hashlib
import json
import math
from pathlib import Path

POLICY_VERSION = "cnc-teaching-v1"
# Illustrative educational floors near the narrative example, not measured
# business requirements or a statistically justified validation sample size.
MIN_EVENTS = 32
LIMITS = {
    "stage1_average_precision": ("min", 0.20),
    "stage1_recall": ("min", 0.90),
    "stage1_flag_rate": ("max", 0.05),
    "cascade_recall": ("min", 0.70),
    "cascade_precision": ("min", 0.20),
    "cascade_flag_rate": ("max", 100 / 3800),
}


def assess(record):
    """Reject malformed or incomplete outcomes, including zero-hold precision."""
    report = {
        "policy_version": POLICY_VERSION,
        "passed": False,
        "production_approval": False,
        "minimum_events": MIN_EVENTS,
        "sample_size_basis": "demonstration floor, not statistical adequacy",
    }
    try:
        n = record["test_rows"]
        c, s = record["combined_test"], record["stage1_test"]
        failures, flags, caught = (
            c[k] for k in ("failures", "flags", "caught")
        )
        if any(
            type(x) is not int
            for x in (n, failures, flags, caught)
        ):
            raise ValueError("integer test counts required")
        if (
            not 0 <= caught <= min(failures, flags) <= n
            or n <= 0
            or failures >= n
            or flags > n
        ):
            raise ValueError("inconsistent binary test counts")
        if failures < MIN_EVENTS:
            raise ValueError(
                "too few events for the teaching gate"
            )
        values = {
            "stage1_average_precision": s["average_precision"],
            "stage1_recall": s["recall"],
            "stage1_flag_rate": s["flag_rate"],
            "cascade_recall": caught / failures,
            "cascade_precision": caught / flags
            if flags
            else None,
            "cascade_flag_rate": flags / n,
        }
        for key, value in values.items():
            if value is None and key == "cascade_precision":
                continue
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(
                    "finite probability metrics required"
                )
        if (
            c.get("recall") != values["cascade_recall"]
            or c.get("precision") != values["cascade_precision"]
        ):
            raise ValueError(
                "reported cascade metrics disagree with counts"
            )
        report["checks"] = [
            dict(
                metric=k,
                value=values[k],
                limit=v,
                relation=op,
                passed=values[k] is not None
                and (
                    values[k] >= v
                    if op == "min"
                    else values[k] <= v
                ),
            )
            for k, (op, v) in LIMITS.items()
        ]
        report["passed"] = all(
            c["passed"] for c in report["checks"]
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
    ) as exc:
        report["error"] = str(exc)
    return report


def evaluate_file(metrics, *, bundle=None):
    try:
        raw = Path(metrics).read_bytes()
        report = assess(json.loads(raw))
        report["metrics_sha256"] = hashlib.sha256(raw).hexdigest()
        if bundle is not None:
            bundle_raw = Path(bundle).read_bytes()
            report["bundle_sha256"] = hashlib.sha256(
                bundle_raw
            ).hexdigest()
            metadata = json.loads(bundle_raw)
            if (
                metadata.get("sha256", {}).get("metrics.json")
                != report["metrics_sha256"]
            ):
                report.update(
                    passed=False,
                    error="bundle does not bind these metrics",
                )
    except (OSError, ValueError, TypeError) as exc:
        report = {
            "policy_version": POLICY_VERSION,
            "passed": False,
            "error": str(exc),
            "production_approval": False,
        }
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metrics", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    a = p.parse_args()
    report = evaluate_file(a.metrics, bundle=a.bundle)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
