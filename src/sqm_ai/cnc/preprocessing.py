"""Persisted numeric feature ordering and category encoding for edge parity."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["part_family", "machine_id", "shift"]
METADATA = [
    "failed",
    "serial",
    "cycle_start",
    "inspection_completed_at",
]


def predictors(frame):
    X = frame.drop(columns=METADATA, errors="ignore").copy()
    if len(X.columns) != 51 or not set(CATEGORICAL).issubset(
        X
    ):
        raise ValueError(
            "expected the 51 named CNC feature columns"
        )
    numeric = [c for c in X if c not in CATEGORICAL]
    if not np.isfinite(
        X[numeric].to_numpy(dtype=float)
    ).all():
        raise ValueError("finite numeric features required")
    if X[CATEGORICAL].isna().any().any():
        raise ValueError("categorical context is missing")
    return X


def fit_preprocessor(X):
    numeric = [c for c in X if c not in CATEGORICAL]
    prep = ColumnTransformer(
        [
            ("numeric", "passthrough", numeric),
            (
                "category",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
        ]
    )
    return prep.fit(X)


def transform(prep, X):
    return np.asarray(prep.transform(X), dtype=np.float32)


def channel_statistics(windows, valid_lengths):
    """Fit normalization on valid TRAINING samples only, excluding padding."""
    total = np.zeros(5, dtype=float)
    squares = np.zeros(5, dtype=float)
    count = 0
    for window, length in zip(
        windows, valid_lengths, strict=True
    ):
        x = np.asarray(window[:, : int(length)], dtype=float)
        if x.shape[1] < 2 or not np.isfinite(x).all():
            raise ValueError(
                "at least two finite valid samples required"
            )
        total += x.sum(1)
        squares += (x * x).sum(1)
        count += x.shape[1]
    if count == 0:
        raise ValueError("no normalization training samples")
    mean = total / count
    scale = np.sqrt(
        np.maximum(squares / count - mean * mean, 0)
    )
    return mean, np.where(scale > 1e-8, scale, 1.0)


def prepare_window(window, mean, scale, n=512):
    """Normalize valid samples, then zero-pad; reject unsupported truncation."""
    x = np.asarray(window, dtype=float)
    if (
        x.ndim != 2
        or x.shape[0] != 5
        or not 2 <= x.shape[1] <= n
    ):
        raise ValueError(
            "expected (5, 2..512) valid samples; longer cycles need a validated policy"
        )
    if not np.isfinite(x).all():
        raise ValueError("sensor samples must be finite")
    mean, scale = np.asarray(mean), np.asarray(scale)
    if (
        mean.shape != (5,)
        or scale.shape != (5,)
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "invalid channel normalization artifact"
        )
    result = np.zeros((5, n), dtype=np.float32)
    result[:, : x.shape[1]] = (x - mean[:, None]) / scale[
        :, None
    ]
    return result
