# src/sqm_ai/ci/gate.py
"""Fail the pipeline when a model is not good enough."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", required=True)
    p.add_argument("--min-auc", type=float, required=True)
    p.add_argument(
        "--min-recall-sev3", type=float, required=True
    )
    a = p.parse_args()

    m = json.loads(Path(a.metrics).read_text())
    checks = [
        ("roc_auc", m["roc_auc"], a.min_auc),
        ("recall_sev3", m["recall_sev3"],
         a.min_recall_sev3),
    ]
    failed = []
    for name, got, floor in checks:
        ok = got >= floor
        mark = "PASS" if ok else "FAIL"
        print(f"{mark} {name}={got:.3f} floor={floor:.2f}")
        if not ok:
            failed.append(name)

    if failed:
        print("quality gate failed:", ", ".join(failed))
        return 1
    print("quality gate passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
