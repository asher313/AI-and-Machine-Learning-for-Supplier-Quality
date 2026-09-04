# src/sqm_ai/stats.py
from collections.abc import Callable

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any statistic."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 2:
        raise ValueError("need at least two observations")
    out = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        out[i] = stat_fn(sample)
    lo = np.percentile(out, 100 * alpha / 2)
    hi = np.percentile(out, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def bootstrap_diff_ci(
    a: np.ndarray,
    b: np.ndarray,
    stat_fn: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """CI for stat_fn(b) - stat_fn(a); resample each group."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        out[i] = stat_fn(sb) - stat_fn(sa)
    lo = np.percentile(out, 100 * alpha / 2)
    hi = np.percentile(out, 100 * (1 - alpha / 2))
    return float(lo), float(hi)
