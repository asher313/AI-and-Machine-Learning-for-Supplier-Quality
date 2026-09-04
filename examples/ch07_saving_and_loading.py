# Chapter 7 — 7.5 Saving and Loading
import joblib

joblib.dump(pipeline, "models/baseline_logreg.joblib")
loaded = joblib.load("models/baseline_logreg.joblib")
assert (loaded.predict_proba(X_test)
        == pipeline.predict_proba(X_test)).all()
