"""Chronological Build 1 training; saves complete fitted pipelines."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from sqm_ai.build1.features import load_modelling_frame
from sqm_ai.features import month_folds, prep

REG_PARAMS = {
    "n_estimators": 1200,
    "learning_rate": 0.04,
    "max_depth": 5,
    "min_child_weight": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": 42,
}
CLF_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.03,
    "max_depth": 6,
    "min_child_weight": 12,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "reg_lambda": 4.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": 42,
}


# SECTION: chronological fitting


def split_train_tail(df, idx, cal_months=3):
    """Whole calibration months; inner labels mature first."""
    idx = np.asarray(idx, dtype=int)
    months = np.sort(df.iloc[idx].month.unique())
    if len(months) <= cal_months:
        raise ValueError("not enough training months")
    selected = months[-cal_months:]
    rows = df.iloc[idx]
    cal = idx[rows.month.isin(selected).to_numpy()]
    first_cutoff = pd.Timestamp(
        selected[0]
    ) + pd.offsets.MonthBegin(1)
    inner = idx[
        (
            rows.label_end.le(first_cutoff)
            & rows.month.lt(selected[0])
        ).to_numpy()
    ]
    if not len(inner) or not len(cal):
        raise ValueError("empty fit or calibration period")
    return inner, cal


def fit_classifier(X, y, df, tr, params=None):
    inner, cal = split_train_tail(df, tr)
    if y.iloc[inner].nunique() != 2 or y.iloc[cal].nunique() != 2:
        raise ValueError("fit and calibration need both classes")
    options = CLF_PARAMS.copy()
    options.update(params or {})
    positives = int(y.iloc[inner].sum())
    options["scale_pos_weight"] = (
        len(inner) - positives
    ) / positives
    fitted = Pipeline(
        [
            ("prep", clone(prep)),
            ("model", xgb.XGBClassifier(**options)),
        ]
    )
    fitted.fit(X.iloc[inner], y.iloc[inner])
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(fitted), method="isotonic"
    )
    calibrated.fit(X.iloc[cal], y.iloc[cal])
    return calibrated


def fit_regressor(X, y, tr, params=None):
    options = REG_PARAMS.copy()
    options.update(params or {})
    model = Pipeline(
        [
            ("prep", clone(prep)),
            ("model", xgb.XGBRegressor(**options)),
        ]
    )
    model.fit(X.iloc[tr], y.iloc[tr])
    return model


# SECTION: evaluation


def run(df, reg_params=None, clf_params=None):
    X, y_clf, y_reg = load_modelling_frame(df)
    rows = []
    for k, (tr, te) in enumerate(month_folds(df), 1):
        first_cutoff = df.iloc[
            te
        ].month.min() + pd.offsets.MonthBegin(1)
        if not df.iloc[tr].label_end.le(first_cutoff).all():
            raise ValueError(
                "training labels unavailable at test cutoff"
            )
        reg = fit_regressor(X, y_reg, tr, reg_params)
        h = np.maximum(0.0, reg.predict(X.iloc[te]))
        clf = fit_classifier(X, y_clf, df, tr, clf_params)
        p = clf.predict_proba(X.iloc[te])[:, 1]
        rows.append(
            {
                "fold": k,
                "n_test": len(te),
                "positive_test": int(y_clf.iloc[te].sum()),
                "screen_threshold": 0.04,
                "recall_sev3": recall_score(
                    y_clf.iloc[te], p >= 0.04
                ),
                "roc_auc": roc_auc_score(y_clf.iloc[te], p),
                "pr_auc": average_precision_score(
                    y_clf.iloc[te], p
                ),
                "brier": brier_score_loss(y_clf.iloc[te], p),
                "mae": mean_absolute_error(y_reg.iloc[te], h),
                "r2": r2_score(y_reg.iloc[te], h),
                "spearman": spearmanr(y_reg.iloc[te], h)[0],
            }
        )
    return pd.DataFrame(rows)


# SECTION: command line and persisted artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/supplier_month.parquet"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/build1")
    )
    parser.add_argument(
        "--trees",
        type=int,
        help="override both tree counts for a smoke run",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("use a new model output directory")
    df = (
        pd.read_parquet(args.input)
        .sort_values(["month", "supplier_id"])
        .reset_index(drop=True)
    )
    override = (
        {} if args.trees is None else {"n_estimators": args.trees}
    )
    if args.trees is not None and args.trees < 1:
        raise ValueError("trees must be positive")
    metrics = run(df, override, override)
    X, yc, yr = load_modelling_frame(df)
    idx = np.arange(len(df))
    reg = fit_regressor(X, yr, idx, override)
    clf = fit_classifier(X, yc, df, idx, override)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(reg, args.output / "risk_regressor.joblib")
    reg.named_steps["model"].save_model(
        args.output / "risk_regressor.json"
    )
    joblib.dump(
        clf, args.output / "risk_classifier_calibrated.joblib"
    )
    (args.output / "feature_list.json").write_text(
        json.dumps(list(X.columns), indent=2) + "\n"
    )
    metrics.to_json(
        args.output / "metrics.json", orient="records", indent=2
    )
    (args.output / "threshold.json").write_text(
        json.dumps(
            {
                "escalate": 0.30,
                "status": "fictional teaching policy; validate before use",
            },
            indent=2,
        )
        + "\n"
    )
    digest = hashlib.sha256(args.input.read_bytes()).hexdigest()
    names = [
        "risk_regressor.joblib",
        "risk_regressor.json",
        "risk_classifier_calibrated.joblib",
        "feature_list.json",
        "threshold.json",
    ]
    artifact_hashes = {
        name: hashlib.sha256(
            (args.output / name).read_bytes()
        ).hexdigest()
        for name in names
    }
    files = [
        Path(__file__),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("score_suppliers.py"),
        Path(__file__).with_name("explain.py"),
        Path(__file__).parent.parent / "features.py",
    ]
    source_hashes = {
        str(
            f.relative_to(Path(__file__).parent.parent)
        ): hashlib.sha256(f.read_bytes()).hexdigest()
        for f in files
    }
    version = hashlib.sha256(
        json.dumps(
            {
                "artifacts": artifact_hashes,
                "source": source_hashes,
                "data": digest,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:12]
    (args.output / "metadata.json").write_text(
        json.dumps(
            {
                "training_data_sha256": digest,
                "training_data_complete_through": str(
                    df.data_complete_through.max().date()
                ),
                "model_version": version,
                "artifact_sha256": artifact_hashes,
                "source_sha256": source_hashes,
                "trees_override": args.trees,
                "intended_use": "Synthetic teaching demonstration; human review only",
            },
            indent=2,
        )
        + "\n"
    )
    print(metrics.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
