# Chapter 7 — 7.2 The Eleven-Step Template
import json
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
)
from sklearn.model_selection import (
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path

from sqm_ai.features import month_folds

# 1. Load
df = pd.read_parquet("data/supplier_month.parquet")

# 2. Separate features from target (and drop the keys)
TARGET = "sev3_next_90d"
KEYS = ["supplier_id", "month"]
X = df.drop(columns=[TARGET] + KEYS)
y = df[TARGET]

# 3. Name the column types explicitly
NUMERIC = [
    "ncr_count_90d", "ncr_per_1k_units", "avg_severity_90d",
    "pct_high_severity", "otd_pct", "avg_days_late", "fpy",
    "fpy_slope_12m", "audit_score_last", "audit_score_delta",
    "years_as_supplier", "spend_share", "car_response_days",
    "car_effectiveness",
]
CATEGORICAL = ["tier"]

# 4. One preprocessing pipeline per column type
numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore",
                             sparse_output=False)),
])

# 5. Route each column set to its pipeline
preprocessor = ColumnTransformer([
    ("num", numeric_pipe, NUMERIC),
    ("cat", categorical_pipe, CATEGORICAL),
])

# 6. Preprocessing + model = one estimator
pipeline = Pipeline([
    ("prep", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1)),
])

# 7. Reserve the latest three months; purge three before them
months = sorted(df["month"].unique())
train = df.loc[df["month"] < months[-6]].copy()
test = df.loc[df["month"] >= months[-3]].copy()
X_train, y_train = train[X.columns], train[TARGET]
X_test, y_test = test[X.columns], test[TARGET]

# 8. Fit
pipeline.fit(X_train, y_train)

# 9. Evaluate on the held-out rows
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
print(confusion_matrix(y_test, y_pred))

# 10. Cross-validate for an honest estimate
cv_auc = cross_val_score(
    pipeline, X_train, y_train,
    cv=list(month_folds(train, n_splits=3)), scoring="roc_auc")
print(f"CV AUC {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")

# 11. Save the whole pipeline, not just the model
Path("models").mkdir(exist_ok=True)
joblib.dump(pipeline, "models/template_rf.joblib")
json.dump({"cv_auc": float(cv_auc.mean())},
          open("models/template_rf_metrics.json", "w"))
