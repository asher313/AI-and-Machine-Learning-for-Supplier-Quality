# src/sqm_ai/cnc/train_stage1.py
"""Chronological stage-1 fitting with a separate threshold period."""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from sqm_ai.cnc.preprocessing import (
    fit_preprocessor,
    predictors,
)

FLAG_RATE = 0.05


def threshold_for_flag_rate(probs, rate):
    """Development quantile; ties and drift can change the achieved rate."""
    probs = np.asarray(probs, dtype=float)
    if (
        not len(probs)
        or not np.isfinite(probs).all()
        or not 0 < rate < 1
    ):
        raise ValueError(
            "finite scores and 0 < rate < 1 required"
        )
    return float(np.quantile(probs, 1 - rate))


def fit_stage1(fit, calibration, trees=500):
    """Fit once; calibration rows set only the gate threshold."""
    fit, calibration = fit.copy(), calibration.copy()
    if fit.empty or calibration.empty:
        raise ValueError(
            "nonempty fit and calibration periods required"
        )
    cutoff = calibration.cycle_start.min()
    if (
        not fit.cycle_start.lt(cutoff).all()
        or not fit.inspection_completed_at.lt(cutoff).all()
    ):
        raise ValueError(
            "fit outcomes must be available before calibration starts"
        )
    X, y = predictors(fit), fit.failed.to_numpy()
    if not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError(
            "binary training data with both classes required"
        )
    prep = fit_preprocessor(X)
    clf = xgb.XGBClassifier(
        n_estimators=trees,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=float((len(y) - y.sum()) / y.sum()),
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
    )
    pipeline = Pipeline([("prep", prep), ("model", clf)])
    pipeline.fit(X, y)
    p = pipeline.predict_proba(predictors(calibration))[:, 1]
    return pipeline, threshold_for_flag_rate(p, FLAG_RATE)


def metrics(y, p, threshold):
    y, p = np.asarray(y), np.asarray(p)
    flags = p >= threshold
    return {
        "average_precision": float(
            average_precision_score(y, p)
        )
        if y.sum()
        else None,
        "flag_rate": float(flags.mean()),
        "recall": float(y[flags].sum() / y.sum())
        if y.sum()
        else None,
        "threshold": float(threshold),
    }


def train_stage1(df, test_size=60_000, trees=500):
    """Five outer folds; the last 20% of each past split calibrates the gate.

    Row-count blocks are valid only after sorting and enforcing timestamps.
    The last fitted model is returned without refitting on its test rows.
    """
    df = df.sort_values("cycle_start").reset_index(drop=True)
    records = []
    model, threshold = None, None
    for fold, (past, test) in enumerate(
        TimeSeriesSplit(
            n_splits=5, test_size=test_size
        ).split(df)
    ):
        test_start = df.iloc[test].cycle_start.min()
        history = df.iloc[past]
        history = history[
            history.inspection_completed_at.lt(test_start)
            & history.cycle_start.lt(test_start)
        ]
        split = int(len(history) * 0.8)
        cutoff = history.iloc[split].cycle_start
        fit = history[
            history.cycle_start.lt(cutoff)
            & history.inspection_completed_at.lt(cutoff)
        ]
        cal = history[history.cycle_start.ge(cutoff)]
        model, threshold = fit_stage1(fit, cal, trees)
        p = model.predict_proba(predictors(df.iloc[test]))[
            :, 1
        ]
        record = {
            "fold": fold,
            **metrics(df.iloc[test].failed, p, threshold),
        }
        records.append(record)
        print(record)
    return model, threshold, pd.DataFrame(records)
