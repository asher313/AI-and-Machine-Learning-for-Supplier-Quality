# Chapter 7 — Encoders
from sklearn.preprocessing import TargetEncoder

# Fit-time uses internal cross-fitting; never call
# fit_transform on rows you will evaluate on.
te = TargetEncoder(target_type="binary", smooth="auto")
