# Chapter 9 — 9.4 Hyperparameter Search
from scipy.stats import randint, uniform
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV,
)

grid = {
    "model__max_depth": [3, 5, 7],
    "model__learning_rate": [0.03, 0.05, 0.1],
    "model__min_child_weight": [1, 5, 20],
}
search = GridSearchCV(
    pipe, grid, cv=folds, scoring="average_precision",
    n_jobs=-1, refit=True, verbose=1,
)

dist = {
    "model__max_depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.15),
    "model__subsample": uniform(0.6, 0.4),
    "model__colsample_bytree": uniform(0.6, 0.4),
    "model__min_child_weight": randint(1, 40),
}
search = RandomizedSearchCV(
    pipe, dist, n_iter=60, cv=folds,
    scoring="average_precision", n_jobs=-1,
    random_state=42,
)
search.fit(X, y)
print(search.best_params_, round(search.best_score_, 3))
