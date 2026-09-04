# Chapter 9 — 9.2 Class Weights, Resampling, and Threshold Tuning
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight="balanced")
model = RandomForestClassifier(class_weight={0: 1, 1: 10})

# XGBoost: one number, not a dict
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos          # 9.4 on this table
