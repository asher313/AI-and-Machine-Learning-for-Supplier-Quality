# src/sqm_ai/build1/score_suppliers.py
"""Nightly entry point: score every active supplier."""
from __future__ import annotations

import json
from datetime import date

import joblib
import pandas as pd
import structlog
import xgboost as xgb

from sqm_ai.build1.explain import explainer_for, top_drivers
from sqm_ai.build1.features import build_features

log = structlog.get_logger()
MODEL_DIR = "models/build1"


def load_artifacts():
    reg = xgb.XGBRegressor()
    reg.load_model(f"{MODEL_DIR}/risk_regressor.json")
    clf = joblib.load(
        f"{MODEL_DIR}/risk_classifier_calibrated.joblib")
    with open(f"{MODEL_DIR}/feature_list.json") as fh:
        cols = json.load(fh)
    with open(f"{MODEL_DIR}/threshold.json") as fh:
        thr = json.load(fh)["escalate"]
    return reg, clf, cols, thr


def score(as_of: date) -> pd.DataFrame:
    reg, clf, cols, thr = load_artifacts()
    feats = build_features(as_of=as_of)
    missing = sorted(set(cols) - set(feats.columns))
    if missing:
        raise ValueError(f"missing features: {missing}")
    X = feats[cols]
    harm = reg.predict(X)
    rank = pd.Series(harm).rank(pct=True) * 100
    out = pd.DataFrame({
        "supplier_id": feats["supplier_id"].to_numpy(),
        "as_of": as_of,
        "predicted_harm_90d": harm.round(2),
        "risk_score": rank.round().astype(int).to_numpy(),
        "p_sev3_90d": clf.predict_proba(X)[:, 1].round(4),
    })
    out["escalate"] = out["p_sev3_90d"] >= thr
    ex = explainer_for(clf)
    out["drivers"] = [
        json.dumps(top_drivers(ex, X, i))
        for i in range(len(X))
    ]
    log.info("scored", n=len(out), as_of=str(as_of),
             escalated=int(out["escalate"].sum()))
    return out
