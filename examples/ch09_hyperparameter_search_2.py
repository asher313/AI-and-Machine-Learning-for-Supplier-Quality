# Chapter 9 — 9.4 Hyperparameter Search
import optuna
import xgboost as xgb
from sklearn.metrics import average_precision_score

from sqm_ai.features import month_folds   # Chapter 7.4


def objective(trial):
    params = {
        "n_estimators": 2000,
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float(
            "subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 1, 40),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", 0.5, 20.0, log=True),
    }
    scores = []
    for tr, te in month_folds(df):
        m = xgb.XGBClassifier(
            **params, eval_metric="aucpr",
            early_stopping_rounds=50, tree_method="hist",
            n_jobs=-1, random_state=42)
        inner = tr[: int(len(tr) * 0.85)]
        hold = tr[int(len(tr) * 0.85) :]
        m.fit(X.loc[inner], y.loc[inner],
              eval_set=[(X.loc[hold], y.loc[hold])],
              verbose=False)
        p = m.predict_proba(X.loc[te])[:, 1]
        scores.append(
            average_precision_score(y.loc[te], p))
    return sum(scores) / len(scores)


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60)
print(study.best_params, round(study.best_value, 3))
