# Chapter 10 — 10.5 Two Models
# src/sqm_ai/build1/train.py
import json

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator  # sklearn>=1.6
from sklearn.metrics import (
    average_precision_score, brier_score_loss,
    mean_absolute_error, r2_score, roc_auc_score,
)

from sqm_ai.build1.features import load_modelling_frame
from sqm_ai.features import month_folds, prep

REG_PARAMS = dict(
    n_estimators=1200, learning_rate=0.04, max_depth=5,
    min_child_weight=8, subsample=0.8,
    colsample_bytree=0.8, reg_lambda=2.0,
    objective="reg:squarederror", tree_method="hist",
    n_jobs=-1, random_state=42,
)
CLF_PARAMS = dict(
    n_estimators=1500, learning_rate=0.03, max_depth=6,
    min_child_weight=12, subsample=0.85,
    colsample_bytree=0.75, reg_lambda=4.0,
    scale_pos_weight=9.4,   # neg/pos at a 0.096 base rate
    objective="binary:logistic", eval_metric="aucpr",
    early_stopping_rounds=50, tree_method="hist",
    n_jobs=-1, random_state=42,
)


# Chapter 10 — 10.5 Two Models (continued)
def split_train_tail(idx, frac=0.15):
    """Last `frac` of a training fold, kept in order."""
    cut = int(len(idx) * (1 - frac))
    return idx[:cut], idx[cut:]


def fit_classifier(X, y, tr):
    inner, cal = split_train_tail(tr)
    clf = xgb.XGBClassifier(**CLF_PARAMS)
    clf.fit(
        prep.fit_transform(X.loc[inner], y.loc[inner]),
        y.loc[inner],
        eval_set=[(prep.transform(X.loc[cal]),
                   y.loc[cal])],
        verbose=False,
    )
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(clf), method="isotonic")
    calibrated.fit(prep.transform(X.loc[cal]), y.loc[cal])
    return calibrated


# Chapter 10 — 10.5 Two Models (continued)
def run(df):
    X, y_clf, y_reg = load_modelling_frame(df)
    rows = []
    for k, (tr, te) in enumerate(month_folds(df), 1):
        reg = xgb.XGBRegressor(**REG_PARAMS)
        reg.fit(prep.fit_transform(X.loc[tr]), y_reg.loc[tr])
        h = reg.predict(prep.transform(X.loc[te]))

        clf = fit_classifier(X, y_clf, tr)
        p = clf.predict_proba(
            prep.transform(X.loc[te]))[:, 1]

        rows.append({
            "fold": k,
            "roc_auc": roc_auc_score(y_clf.loc[te], p),
            "pr_auc": average_precision_score(
                y_clf.loc[te], p),
            "brier": brier_score_loss(y_clf.loc[te], p),
            "mae": mean_absolute_error(y_reg.loc[te], h),
            "r2": r2_score(y_reg.loc[te], h),
            "spearman": spearmanr(y_reg.loc[te], h)[0],
        })
    return pd.DataFrame(rows).round(3)
