# Chapter 9 — 9.5 Feature Engineering Patterns
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures

# 1. Interactions: every pairwise product
poly = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False)
X_pairs = poly.fit_transform(X[["fpy", "otd_pct"]])

# 2. Domain ratios and trends (the ones that pay)
df["ncr_per_1k_units"] = (
    1000 * df["ncr_count_90d"] / df["units_received_90d"])
df["fpy_delta_3m"] = (
    df["fpy"] - df.groupby("supplier_id")["fpy"].shift(3))
df["years_as_supplier"] = (
    (df["month"] - df["onboarded_at"]).dt.days / 365.25)

# 3. Cross-fitted target encoding (leak-safe)
def cv_target_encode(frame, col, target, n_splits=5):
    out = np.zeros(len(frame))
    kf = KFold(n_splits=n_splits, shuffle=True,
               random_state=42)
    prior = frame[target].mean()
    for tr, va in kf.split(frame):
        means = frame.iloc[tr].groupby(col)[target].mean()
        out[va] = (frame.iloc[va][col].map(means)
                   .fillna(prior).to_numpy())
    return out

# 4. Date parts and elapsed time
df["month_of_year"] = df["month"].dt.month
df["days_since_audit"] = (
    df["month"] - df["audit_date_last"]).dt.days
