"""Explain predicted severity-sum harm, before percentile ranking."""

import numpy as np
import pandas as pd
import shap

READABLE = {
    "ncr_per_1k_units": "nonconformances per 1,000 units",
    "pct_high_severity": "share of NCRs at severity 3+",
    "avg_severity_90d": "average severity, last 90 days",
    "audit_score_delta": "change in audit score",
    "audit_score_last": "most recent audit score",
    "fpy_slope_12m": "first-pass-yield trend, 12 months",
    "otd_pct": "on-time delivery, last 90 days",
    "avg_days_late": "average days late",
    "car_response_days": "days to respond to a CAR",
    "car_effectiveness": "CAR effectiveness rate",
    "years_as_supplier": "years as a supplier",
    "spend_share": "share of total spend",
    "tier": "supplier tier",
    "fpy": "first-pass yield",
}


def explainer_for(regressor):
    """TreeExplainer for the fitted harm regressor's booster."""
    return shap.TreeExplainer(regressor.named_steps["model"])


def transformed_frame(regressor, X):
    prep = regressor.named_steps["prep"]
    return pd.DataFrame(
        prep.transform(X),
        columns=prep.get_feature_names_out(),
        index=X.index,
    )


def top_drivers(explainer, X, i, k=5, raw=None, values=None):
    """X is transformed; raw retains human-readable units."""
    sv = (
        explainer.shap_values(X.iloc[[i]])[0]
        if values is None
        else values
    )
    order = np.argsort(-np.abs(sv))[:k]
    result = []
    for j in order:
        name = X.columns[j].split("__", 1)[-1]
        value = (
            raw.iloc[i][name]
            if raw is not None and name in raw.columns
            else X.iloc[i, j]
        )
        if pd.isna(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        result.append(
            {
                "feature": name,
                "label": READABLE.get(name, name),
                "value": value,
                "impact": float(sv[j]),
                "impact_units": "raw predicted harm before zero floor",
                "direction": "increases predicted harm"
                if sv[j] > 0
                else "decreases predicted harm"
                if sv[j] < 0
                else "no contribution",
            }
        )
    return result
