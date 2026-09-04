# Chapter 9 — 9.2 Class Weights, Resampling, and Threshold Tuning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

pipe = ImbPipeline([
    ("prep", prep),
    ("sample", SMOTE(random_state=42)),
    ("model", RandomForestClassifier(random_state=42)),
])
