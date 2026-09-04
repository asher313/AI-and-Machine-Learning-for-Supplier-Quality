# Chapter 7 — 7.1 The Five Methods Every Estimator Has
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000)),
])
pipe.fit(X_train, y_train)     # scale.fit_transform -> model.fit
pipe.predict(X_test)           # scale.transform -> model.predict
