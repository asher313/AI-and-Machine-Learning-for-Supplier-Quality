# sqm_ai/cnc/train_stage1.py
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

FLAG_RATE = 0.05


def threshold_for_flag_rate(
    probs: np.ndarray, rate: float
) -> float:
    """Threshold that flags exactly `rate` of cycles."""
    return float(np.quantile(probs, 1.0 - rate))


def train_stage1(
    X: pd.DataFrame, y: np.ndarray
) -> tuple[xgb.XGBClassifier, float]:
    """Fit stage 1. X must be sorted by cycle start."""
    pos = float(y.sum())
    pos_weight = (len(y) - pos) / pos     # 165.7
    cv = TimeSeriesSplit(n_splits=5, test_size=60_000)
    model = None
    for fold, (tr, va) in enumerate(cv.split(X)):
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            early_stopping_rounds=20,
            tree_method="hist",
            enable_categorical=True,
        )
        model.fit(
            X.iloc[tr], y[tr],
            eval_set=[(X.iloc[va], y[va])],
            verbose=False,
        )
        p = model.predict_proba(X.iloc[va])[:, 1]
        ap = average_precision_score(y[va], p)
        thr = threshold_for_flag_rate(p, FLAG_RATE)
        recall = float(y[va][p >= thr].sum() / y[va].sum())
        print(
            f"fold {fold} AUC-PR={ap:.4f} "
            f"thr={thr:.4f} recall@5%={recall:.3f}"
        )
    # the shipped threshold comes from the last fold's
    # held-out cycles, never from rows used in training
    last_p = model.predict_proba(X.iloc[va])[:, 1]
    return model, threshold_for_flag_rate(last_p, FLAG_RATE)
