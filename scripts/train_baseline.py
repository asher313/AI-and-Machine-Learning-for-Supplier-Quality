# scripts/train_baseline.py
# Chapter 7 — Asher at Northlake
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# The book's listing says "(NUMERIC, TARGET, KEYS and
# month_folds as defined above)"; in the repository they
# live in sqm_ai.features (Chapter 7.2 and 7.4).
from sqm_ai.features import KEYS, NUMERIC, TARGET, month_folds
df = pd.read_parquet("data/supplier_month.parquet")
df = df.sort_values(["month", "supplier_id"]).reset_index(
    drop=True)
X = df.drop(columns=[TARGET] + KEYS)
y = df[TARGET]

prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median",
                                 add_indicator=True)),
        ("scale", StandardScaler()),
    ]), NUMERIC),
    ("tier", OrdinalEncoder(categories=[["C", "B", "A"]],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1), ["tier"]),
])
baseline = Pipeline([
    ("prep", prep),
    ("model", LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000)),
])

aucs, aps = [], []
for tr, te in month_folds(df):
    baseline.fit(X.loc[tr], y.loc[tr])
    p = baseline.predict_proba(X.loc[te])[:, 1]
    aucs.append(roc_auc_score(y.loc[te], p))
    aps.append(average_precision_score(y.loc[te], p))
print("ROC-AUC per fold:", [round(a, 3) for a in aucs])
print("PR-AUC  per fold:", [round(a, 3) for a in aps])
