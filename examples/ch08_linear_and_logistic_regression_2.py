# Chapter 8 — 8.1 Linear and Logistic Regression
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    C=1.0,                   # INVERSE strength: small C = more
    penalty="l2",            # "l1", "l2", "elasticnet", None
    solver="lbfgs",          # "saga" for L1 or elasticnet
    max_iter=2000,
    class_weight="balanced", # see Chapter 9
    random_state=42,
)
