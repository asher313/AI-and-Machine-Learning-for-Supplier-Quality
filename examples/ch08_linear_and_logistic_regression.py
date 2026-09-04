# Chapter 8 — 8.1 Linear and Logistic Regression
from sklearn.linear_model import (
    ElasticNet, Lasso, LinearRegression, Ridge,
)

model = LinearRegression()          # no penalty
model = Ridge(alpha=1.0)            # L2; alpha = strength
model = Lasso(alpha=0.1)            # L1; zeroes coefficients
model = ElasticNet(alpha=0.1, l1_ratio=0.5)

model.coef_                         # one weight per feature
model.intercept_                    # the constant term
