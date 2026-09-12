"""Offline synthetic Build 2 walkthrough with sequential development periods."""

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from sqm_ai.cnc.export import (
    assert_parity,
    assert_stage1_parity,
    export_stage1,
    export_stage2,
)
from sqm_ai.cnc.preprocessing import (
    predictors,
    prepare_window,
    transform,
)
from sqm_ai.cnc.train_stage1 import fit_stage1, metrics
from sqm_ai.cnc.train_stage2 import fit_stage2


def run(data, output, trees=500, epochs=30):
    data, output = Path(data), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use an empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(
        data / "cycle_features.parquet"
    ).sort_values("cycle_start")
    pilot = pd.read_parquet(data / "pilot_cycles.parquet")
    windows = np.load(data / "windows.npy", mmap_mode="r")
    lengths = np.load(data / "valid_lengths.npy")
    labels = np.load(data / "labels.npy")
    if len(pilot) != len(windows) or not np.array_equal(
        labels, pilot.failed
    ):
        raise ValueError(
            "pilot metadata/windows/labels are not aligned"
        )
    if (
        not pilot.cycle_start.is_monotonic_increasing
        or pilot.serial.duplicated().any()
    ):
        raise ValueError(
            "unique chronological pilot rows required"
        )
    n = len(pilot)
    cut = [
        pilot.cycle_start.iloc[int(n * q)]
        for q in [0.4, 0.5, 0.7, 0.85]
    ]
    a, b, c, d = cut
    fit = df[
        df.cycle_start.lt(a)
        & df.inspection_completed_at.lt(a)
    ]
    cal = df[
        df.cycle_start.ge(a)
        & df.cycle_start.lt(b)
        & df.inspection_completed_at.lt(b)
    ]
    stage1, threshold = fit_stage1(fit, cal, trees)
    # Gate is frozen before any CNN development examples occur.
    aligned = (
        df.set_index("serial").loc[pilot.serial].reset_index()
    )
    p1 = stage1.predict_proba(predictors(aligned))[:, 1]
    gated = p1 >= threshold
    tr = np.flatnonzero(
        gated
        & pilot.cycle_start.ge(b)
        & pilot.cycle_start.lt(c)
        & pilot.inspection_completed_at.lt(c)
    )
    va = np.flatnonzero(
        gated
        & pilot.cycle_start.ge(c)
        & pilot.cycle_start.lt(d)
        & pilot.inspection_completed_at.lt(d)
    )
    te = np.flatnonzero(pilot.cycle_start.ge(d))
    if not len(tr) or not len(va) or not len(te):
        raise ValueError(
            "gate produced insufficient chronological partitions"
        )
    dev = np.r_[tr, va]
    cnn, mean, scale = fit_stage2(
        windows[dev],
        labels[dev],
        len(tr),
        lengths[dev],
        epochs=epochs,
        ckpt_path=str(output / "stage2.pt"),
    )

    # A validation-only cost sweep; scores are not claimed calibrated.
    def cnn_scores(indices):
        prepared = np.stack(
            [
                prepare_window(
                    windows[i, :, : int(lengths[i])],
                    mean,
                    scale,
                )
                for i in indices
            ]
        )
        with torch.no_grad():
            p = expit(cnn(torch.from_numpy(prepared)).numpy())
        return p, prepared

    p_val, _ = cnn_scores(va)
    candidates = np.r_[
        0.0, np.unique(p_val), np.nextafter(1.0, 2.0)
    ]
    costs = [
        85 * np.sum((p_val >= t) & (labels[va] == 0))
        + 4100 * np.sum((p_val < t) & (labels[va] == 1))
        for t in candidates
    ]
    val_population = int(
        (
            pilot.cycle_start.ge(c)
            & pilot.cycle_start.lt(d)
            & pilot.inspection_completed_at.lt(d)
        ).sum()
    )
    capacity = (
        100 / 3800
    )  # illustrative maximum population flag fraction
    costs = [
        cost
        if np.sum(p_val >= t) / val_population <= capacity
        else float("inf")
        for t, cost in zip(candidates, costs, strict=True)
    ]
    t2 = float(candidates[int(np.argmin(costs))])
    p_test, prepared = cnn_scores(te)
    holds = gated[te] & (p_test >= t2)
    actual = labels[te]
    record = {
        "synthetic": True,
        "scope": "single-cell pilot test; not an all-cell production validation",
        "period_cutoffs": [str(t) for t in cut],
        "stage1_fit_rows": len(fit),
        "stage1_calibration_rows": len(cal),
        "stage2_fit_rows": len(tr),
        "stage2_validation_rows": len(va),
        "test_rows": len(te),
        "validation_flag_fraction_limit": capacity,
        "stage1_test": metrics(actual, p1[te], threshold),
        "combined_test": {
            "flags": int(holds.sum()),
            "failures": int(actual.sum()),
            "caught": int(actual[holds].sum()),
            "recall": float(
                actual[holds].sum() / actual.sum()
            )
            if actual.sum()
            else None,
            "precision": float(actual[holds].mean())
            if holds.sum()
            else None,
            "error_cost": int(
                85 * np.sum(holds & (actual == 0))
                + 4100 * np.sum(~holds & (actual == 1))
            ),
        },
    }
    record["versions"] = {
        name: version(name)
        for name in [
            "numpy",
            "pandas",
            "scikit-learn",
            "xgboost",
            "torch",
            "onnx",
            "onnxruntime",
            "onnxmltools",
        ]
    }
    record["generator_manifest_sha256"] = hashlib.sha256(
        (data / "manifest.json").read_bytes()
    ).hexdigest()
    sources = [
        Path(__file__),
        Path(__file__).with_name("preprocessing.py"),
        Path(__file__).with_name("train_stage1.py"),
        Path(__file__).with_name("train_stage2.py"),
        Path(__file__).with_name("edge.py"),
        Path(__file__).with_name("export.py"),
        Path(__file__).parent.parent / "dl/train.py",
        Path(__file__).parent.parent / "dl/cnc_cnn.py",
    ]
    record["source_sha256"] = {
        str(
            p.relative_to(Path(__file__).parent.parent)
        ): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sources
    }
    stage1.named_steps["model"].save_model(
        output / "stage1_model.json"
    )
    joblib.dump(
        stage1.named_steps["prep"],
        output / "preprocessing.joblib",
    )
    numeric = transform(
        stage1.named_steps["prep"],
        predictors(aligned.iloc[te[:32]]),
    )
    export_stage1(
        stage1.named_steps["model"],
        numeric.shape[1],
        str(output / "stage1.onnx"),
    )
    export_stage2(cnn, str(output / "stage2.onnx"))
    assert_stage1_parity(
        stage1.named_steps["model"],
        output / "stage1.onnx",
        numeric,
    )
    for size in [1, min(7, len(prepared))]:
        assert_parity(
            cnn, output / "stage2.onnx", prepared[:size]
        )
    (output / "metrics.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir()
        if p.is_file()
    }
    config = {
        "stage1_threshold": threshold,
        "stage2_threshold": t2,
        "channel_mean": mean.tolist(),
        "channel_scale": scale.tolist(),
        "sha256": hashes,
    }
    config["model_version"] = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]
    (output / "bundle.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    # End-to-end: raw held-out rows must match in-process decisions.
    from sqm_ai.cnc.edge import CycleScorer

    scorer = CycleScorer(output)
    checks = np.unique(
        np.r_[
            np.arange(min(16, len(te))),
            np.flatnonzero(gated[te])[:8],
        ]
    )
    for j in checks:
        i = te[j]
        row = aligned.iloc[i]
        context = {
            "nominal_seconds": row.cycle_seconds
            / row.cycle_vs_nominal,
            "tool_life_pct": row.tool_life_pct,
            "prior5_fails": row.prior5_fail_count,
            "part_family": row.part_family,
            "machine_id": row.machine_id,
            "shift": row["shift"],
        }
        result = scorer.score(
            windows[i, :, : int(lengths[i])], context
        )
        assert np.isclose(
            result["stage1_score"], p1[i], atol=1e-5
        )
        assert (result["decision"] == "qa_hold") == bool(
            holds[j]
        )
    print(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/cnc")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/build2"),
    )
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    run(args.data, args.output, args.trees, args.epochs)


if __name__ == "__main__":
    main()
