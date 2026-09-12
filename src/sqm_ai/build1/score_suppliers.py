"""Score a versioned monthly snapshot; all recommendations need review."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sqm_ai.build1.explain import (
    explainer_for,
    top_drivers,
    transformed_frame,
)
from sqm_ai.build1.features import build_features

MODEL_DIR = "models/build1"


def load_artifacts(model_dir=MODEL_DIR):
    folder = Path(model_dir)
    # Verify file consistency before loading trusted model artifacts.
    metadata = json.loads((folder / "metadata.json").read_text())
    for name, expected in metadata["artifact_sha256"].items():
        actual = hashlib.sha256(
            (folder / name).read_bytes()
        ).hexdigest()
        if actual != expected:
            raise ValueError(
                f"model artifact checksum mismatch: {name}"
            )
    reg = joblib.load(folder / "risk_regressor.joblib")
    clf = joblib.load(
        folder / "risk_classifier_calibrated.joblib"
    )
    cols = json.loads((folder / "feature_list.json").read_text())
    thr = json.loads((folder / "threshold.json").read_text())[
        "escalate"
    ]
    if not 0 <= thr <= 1:
        raise ValueError("invalid probability threshold")
    return reg, clf, cols, thr, metadata


def score(
    as_of: date,
    model_dir=MODEL_DIR,
    data_path="data/supplier_month_all.parquet",
):
    reg, clf, cols, thr, meta = load_artifacts(model_dir)
    as_of = pd.Timestamp(as_of)
    if as_of < pd.Timestamp(
        meta["training_data_complete_through"]
    ):
        raise ValueError(
            "model uses data unavailable on scoring date"
        )
    feats = build_features(as_of=as_of, data_path=data_path)
    missing = sorted(set(cols) - set(feats.columns))
    if missing:
        raise ValueError(f"missing features: {missing}")
    X = feats[cols]
    raw_harm = reg.predict(X)
    harm = np.maximum(0.0, raw_harm)
    probabilities = clf.predict_proba(X)[:, 1]
    if (
        not np.isfinite(raw_harm).all()
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ValueError("invalid predictions")
    # Midrank percentile in [0, 100]; tied suppliers stay tied.
    rank = (
        100
        * (pd.Series(harm).rank(method="average") - 0.5)
        / len(harm)
    )
    if not feats["limited_history"].isin([True, False]).all():
        raise ValueError(
            "limited_history must be a nonmissing boolean"
        )
    limited = feats["limited_history"].astype(bool).to_numpy()
    inactive = (
        feats.ncr_count_90d.fillna(0).eq(0)
        & feats.units_received_90d.fillna(0).le(0)
    ).to_numpy()
    out = pd.DataFrame(
        {
            "supplier_id": feats.supplier_id.to_numpy(),
            "as_of": as_of,
            "snapshot_cutoff": feats.snapshot_cutoff.to_numpy(),
            "forecast_start": feats.snapshot_cutoff.to_numpy(),
            "forecast_end": (
                feats.snapshot_cutoff + pd.Timedelta(days=90)
            ).to_numpy(),
            "model_version": meta["model_version"],
            "raw_predicted_harm_90d": raw_harm,
            "predicted_harm_90d": harm,
            "harm_was_clipped": raw_harm < 0,
            "risk_score": rank.to_numpy(),
            "p_sev3_90d": probabilities,
            "limited_history": limited,
            "insufficient_recent_activity": inactive,
            "escalate": (probabilities >= thr)
            & ~limited
            & ~inactive,
            "intended_use": "Audit planning; human review required",
        }
    )
    ex = explainer_for(reg)
    transformed = transformed_frame(reg, X)
    values = ex.shap_values(transformed)
    out["drivers"] = [
        json.dumps(
            top_drivers(
                ex, transformed, i, raw=X, values=values[i]
            ),
            allow_nan=False,
        )
        for i in range(len(X))
    ]
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--as-of", type=date.fromisoformat, required=True
    )
    parser.add_argument("--models", default=MODEL_DIR)
    parser.add_argument(
        "--data", default="data/supplier_month_all.parquet"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/supplier_risk_scores.parquet"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("use a new score output path")
    result = score(args.as_of, args.models, args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(
        result.drop(columns=["drivers"])
        .head()
        .to_string(index=False)
    )
    print(f"Wrote {len(result)} suppliers to {args.output}")


if __name__ == "__main__":
    main()
