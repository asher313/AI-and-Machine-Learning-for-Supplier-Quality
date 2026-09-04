# src/sqm_ai/build1/explain.py
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


def explainer_for(model):
    """TreeExplainer over the calibrated model's base."""
    base = model.calibrated_classifiers_[0].estimator
    return shap.TreeExplainer(base)


def top_drivers(explainer, X, i, k=5):
    sv = explainer.shap_values(X.iloc[[i]])[0]
    order = sorted(range(len(sv)),
                   key=lambda j: abs(sv[j]),
                   reverse=True)[:k]
    return [
        {
            "feature": X.columns[j],
            "label": READABLE.get(X.columns[j],
                                  X.columns[j]),
            "value": float(X.iloc[i, j]),
            "impact": round(float(sv[j]), 3),
            "direction": ("increases risk" if sv[j] > 0
                          else "decreases risk"),
        }
        for j in order
    ]
