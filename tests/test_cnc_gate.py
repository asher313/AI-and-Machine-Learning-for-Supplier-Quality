import hashlib
import json
from pathlib import Path
import re
import pytest

from sqm_ai.cnc.gate import assess, evaluate_file


def passing():
    return {
        "test_rows": 10000,
        "stage1_test": {
            "average_precision": 0.25,
            "recall": 0.94,
            "flag_rate": 0.05,
        },
        "combined_test": {
            "failures": 100,
            "flags": 200,
            "caught": 72,
            "recall": 0.72,
            "precision": 0.36,
        },
    }


def test_cascade_reject_all_cannot_hide_behind_stage1():
    record = passing()
    assert assess(record)["passed"]
    record["combined_test"].update(
        flags=0, caught=0, recall=0.0, precision=None
    )
    report = assess(record)
    assert not report["passed"]
    assert not next(
        c
        for c in report["checks"]
        if c["metric"] == "cascade_precision"
    )["passed"]


def test_gate_rejects_inconsistent_counts_and_binds_bytes(
    tmp_path,
):
    record = passing()
    record["combined_test"]["caught"] = 300
    assert not assess(record)["passed"]
    record = passing()
    record["stage1_test"]["average_precision"] = float("nan")
    assert not assess(record)["passed"]
    record = passing()
    record["combined_test"]["recall"] = 0.99
    assert not assess(record)["passed"]
    raw = json.dumps(passing()).encode()
    path = tmp_path / "metrics.json"
    path.write_bytes(raw)
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "sha256": {
                    "metrics.json": hashlib.sha256(
                        raw
                    ).hexdigest()
                }
            }
        )
    )
    report = evaluate_file(path, bundle=bundle)
    assert (
        report["passed"]
        and report["metrics_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
    assert (
        report["bundle_sha256"]
        == hashlib.sha256(bundle.read_bytes()).hexdigest()
    )
    bundle.write_text("{}")
    assert not evaluate_file(path, bundle=bundle)["passed"]
    assert not evaluate_file(tmp_path / "missing.json")["passed"]


def test_recorded_negative_default_run_fails_gate():
    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "docs/synthetic_build2_run.json").read_text()
    )
    assert not assess(record)["passed"]


def test_scorer_rejects_failed_metrics_before_loading_models(
    tmp_path,
):
    pytest.importorskip("onnxruntime")
    from sqm_ai.cnc.edge import CycleScorer

    record = passing()
    record["combined_test"].update(
        flags=0, caught=0, recall=0.0, precision=None
    )
    raw = json.dumps(record).encode()
    (tmp_path / "metrics.json").write_bytes(raw)
    (tmp_path / "bundle.json").write_text(
        json.dumps(
            {
                "sha256": {
                    "metrics.json": hashlib.sha256(
                        raw
                    ).hexdigest()
                }
            }
        )
    )
    with pytest.raises(
        ValueError, match="predictive acceptance failed"
    ):
        CycleScorer(tmp_path)


def test_chapter_map_references_existing_files():
    root = Path(__file__).resolve().parents[1]
    paths = re.findall(
        r"^\| `([^`]+)`",
        (root / "docs/CHAPTER_MAP.md").read_text(),
        re.M,
    )
    assert paths
    assert [p for p in paths if not (root / p).is_file()] == []
