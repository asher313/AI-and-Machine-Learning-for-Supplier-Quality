# Chapter 14 — 14.4 Features From Signals
# sqm_ai/cnc/features.py
import numpy as np
from scipy import stats

SENSORS = [
    "spindle_speed", "feed_rate", "vibration",
    "tool_temp", "coolant_flow",
]
FS = 10.0                       # sample rate, Hz
BANDS = [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0)]


def aggregates(x: np.ndarray, name: str) -> dict:
    """Six statistics for one channel."""
    return {
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
        f"{name}_min": float(np.min(x)),
        f"{name}_max": float(np.max(x)),
        f"{name}_skew": float(stats.skew(x)),
        f"{name}_kurt": float(stats.kurtosis(x)),
    }


def band_energy(x: np.ndarray, name: str) -> dict:
    """Share of spectral energy in each frequency band."""
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), d=1.0 / FS)
    total = float(spec.sum()) + 1e-9
    out = {}
    for lo, hi in BANDS:
        sel = (freq >= lo) & (freq < hi)
        out[f"{name}_band_{lo}_{hi}"] = (
            float(spec[sel].sum()) / total
        )
    hi_key = f"{name}_band_{BANDS[-1][0]}_{BANDS[-1][1]}"
    lo_key = f"{name}_band_{BANDS[0][0]}_{BANDS[0][1]}"
    out[f"{name}_band_ratio"] = (
        out[hi_key] / (out[lo_key] + 1e-9)
    )
    return out


def temporal(x: np.ndarray, name: str, hi: float) -> dict:
    """How long and how often the channel misbehaved."""
    above = x > hi
    runs, current = [], 0
    for flag in above:
        if flag:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return {
        f"{name}_frac_above": float(above.mean()),
        f"{name}_longest_run_s": (
            float(max(runs)) / FS if runs else 0.0
        ),
        f"{name}_excursions": float(len(runs)),
    }


# Chapter 14 — 14.4 Features From Signals (continued)
# sqm_ai/cnc/features.py (continued)
import pandas as pd

THRESHOLDS = {
    "vibration": 2.5,           # g, plant-set alarm level
    "tool_temp": 180.0,         # degrees C
}


def cycle_features(
    window: np.ndarray, context: dict
) -> dict:
    """window: (5, n) float array; context: cycle metadata."""
    if window.shape[0] != len(SENSORS):
        raise ValueError("expected 5 sensor channels")
    feats: dict[str, float] = {}
    for i, name in enumerate(SENSORS):
        channel = window[i]
        feats.update(aggregates(channel, name))
        if name in ("vibration", "tool_temp"):
            feats.update(band_energy(channel, name))
            feats.update(
                temporal(channel, name, THRESHOLDS[name])
            )
    feats["cycle_seconds"] = window.shape[1] / FS
    feats["cycle_vs_nominal"] = (
        feats["cycle_seconds"] / context["nominal_seconds"]
    )
    feats["tool_life_pct"] = context["tool_life_pct"]
    feats["prior5_fail_count"] = context["prior5_fails"]
    feats["part_family"] = context["part_family"]
    feats["machine_id"] = context["machine_id"]
    feats["shift"] = context["shift"]
    return feats


def frame(rows: list[dict]) -> pd.DataFrame:
    """Feature rows -> a model-ready DataFrame."""
    df = pd.DataFrame(rows)
    for col in ("part_family", "machine_id", "shift"):
        df[col] = df[col].astype("category")
    return df
