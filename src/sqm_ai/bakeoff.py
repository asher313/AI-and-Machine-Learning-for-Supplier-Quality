# src/sqm_ai/bakeoff.py
import time

import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from sqm_ai.features import month_folds, prep  # Chapter 7


def evaluate(name, model, df, X, y):
    """One fair row: mean and last-fold scores."""
    aucs, aps = [], []
    t0 = time.perf_counter()
    pipe = Pipeline([("prep", prep), ("model", model)])
    for tr, te in month_folds(df):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        p = pipe.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y.iloc[te], p))
        aps.append(average_precision_score(y.iloc[te], p))
    return {
        "model": name,
        "roc_auc": round(sum(aucs) / len(aucs), 3),
        "roc_auc_last": round(aucs[-1], 3),
        "pr_auc": round(sum(aps) / len(aps), 3),
        "seconds": round(time.perf_counter() - t0, 1),
    }


CONTENDERS = {
    "logreg (Ch 7)": LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=2000),
    "tree d6": DecisionTreeClassifier(
        max_depth=6, min_samples_leaf=25,
        class_weight="balanced", random_state=42),
    "random forest": RandomForestClassifier(
        n_estimators=500, max_features="sqrt", n_jobs=-1,
        class_weight="balanced_subsample", random_state=42),
    "sklearn hist gbm": HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.05, random_state=42,
        early_stopping=False),
    "xgboost": xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", tree_method="hist",
        n_jobs=-1, random_state=42),
    "lightgbm": lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=31,
        min_child_samples=25, n_jobs=-1, random_state=42,
        verbose=-1),
    "knn k=25": KNeighborsClassifier(
        n_neighbors=25, weights="distance", n_jobs=-1),
}

if __name__ == "__main__":
    df = pd.read_parquet("data/supplier_month.parquet")
    df = df.sort_values(["month", "supplier_id"])
    df = df.reset_index(drop=True)
    X = df.drop(columns=["sev3_next_90d", "supplier_id",
                         "month"])
    y = df["sev3_next_90d"]
    rows = [evaluate(n, m, df, X, y)
            for n, m in CONTENDERS.items()]
    out = pd.DataFrame(rows).sort_values(
        "roc_auc", ascending=False)
    print(out.to_string(index=False))
