# Chapter 7 — Scalers
from sklearn.preprocessing import (
    MinMaxScaler, QuantileTransformer, RobustScaler,
    StandardScaler,
)

scale = StandardScaler()             # mean 0, std 1
scale = RobustScaler()               # median 0, IQR 1
scale = MinMaxScaler((0, 1))         # bounded
scale = QuantileTransformer(
    output_distribution="normal", n_quantiles=200)
