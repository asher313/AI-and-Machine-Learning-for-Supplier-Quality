"""Generate streamed CNC fixtures from simulated sensor windows.

No trained predictions or desired model metrics enter this generator.
Run: python -m sqm_ai.cnc.synthetic --output data/cnc
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from sqm_ai.cnc.features import BANDS, FS, SENSORS, THRESHOLDS


def batch_features(windows: np.ndarray, context: pd.DataFrame):
    """Vectorized equivalent of cycle_features for equal lengths."""
    windows = np.asarray(windows, dtype=np.float64)
    if (
        windows.ndim != 3
        or windows.shape[1] != 5
        or windows.shape[2] < 2
        or not np.isfinite(windows).all()
    ):
        raise ValueError("need finite (cycles, 5, samples>=2)")
    if len(context) != len(windows):
        raise ValueError("one context row per window required")
    feats = {}
    for channel, name in enumerate(SENSORS):
        x = windows[:, channel, :]
        feats[name + "_mean"] = x.mean(axis=1)
        feats[name + "_std"] = x.std(axis=1)
        feats[name + "_min"] = x.min(axis=1)
        feats[name + "_max"] = x.max(axis=1)
        variable = x.std(axis=1) > 0
        feats[name + "_skew"] = np.zeros(len(x))
        feats[name + "_kurt"] = np.zeros(len(x))
        feats[name + "_skew"][variable] = stats.skew(
            x[variable], axis=1
        )
        feats[name + "_kurt"][variable] = stats.kurtosis(
            x[variable], axis=1
        )
        if name not in THRESHOLDS:
            continue
        spec = (
            np.abs(np.fft.rfft(x - x.mean(axis=1, keepdims=True)))
            ** 2
        )
        freq = np.fft.rfftfreq(x.shape[1], d=1 / FS)
        total = spec.sum(axis=1) + 1e-9
        for k, (lo, hi) in enumerate(BANDS):
            selected = (freq >= lo) & (
                (freq <= hi)
                if k == len(BANDS) - 1
                else (freq < hi)
            )
            feats[f"{name}_band_{lo}_{hi}"] = (
                spec[:, selected].sum(axis=1) / total
            )
        low = feats[f"{name}_band_{BANDS[0][0]}_{BANDS[0][1]}"]
        high = feats[f"{name}_band_{BANDS[-1][0]}_{BANDS[-1][1]}"]
        feats[name + "_band_ratio"] = high / (low + 1e-9)
        above = x > THRESHOLDS[name]
        feats[name + "_frac_above"] = above.mean(axis=1)
        feats[name + "_excursions"] = (
            above[:, 0].astype(int)
            + (above[:, 1:] & ~above[:, :-1]).sum(axis=1)
        ).astype(float)
        run = np.zeros(len(x), dtype=int)
        longest = run.copy()
        for col in above.T:
            run = np.where(col, run + 1, 0)
            longest = np.maximum(longest, run)
        feats[name + "_longest_run_s"] = longest / FS
    feats["cycle_seconds"] = np.full(
        len(windows), windows.shape[2] / FS
    )
    feats["cycle_vs_nominal"] = (
        feats["cycle_seconds"]
        / context.nominal_seconds.to_numpy()
    )
    feats["tool_life_pct"] = context.tool_life_pct.to_numpy()
    feats["prior5_fail_count"] = context.prior5_fails.to_numpy()
    for col in ["part_family", "machine_id", "shift"]:
        feats[col] = context[col].to_numpy()
    return pd.DataFrame(feats)


def generate(
    output: Path, cycles=931240, raw_cycles=40000, batch_size=512
) -> dict:
    if cycles < 1000 or batch_size < 1 or raw_cycles < 0:
        raise ValueError("cycles >= 1000; batch > 0; raw >= 0")
    raw_cycles = min(raw_cycles, cycles)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("use an empty CNC output directory")
    rng = np.random.default_rng(20260914)
    idx = np.arange(cycles)
    raw_count = 971000 if cycles == 931240 else cycles
    source_idx = np.floor(idx * raw_count / cycles).astype(int)
    raw_positions = np.flatnonzero(source_idx % 14 == 6)[
        :raw_cycles
    ]
    raw_cycles = len(raw_positions)
    # Latent wear drives both signals and failure propensity;
    # unobserved variation prevents a deterministic signal label.
    wear = (idx // 14 % 180) / 180
    propensity = 2.5 * wear + rng.gumbel(size=cycles)
    failed = np.zeros(cycles, dtype=np.int8)
    n_fail = 5587 if cycles == 931240 else round(0.006 * cycles)
    # The pilot fixture has exactly 240 / 40,000 failures.
    pilot_fail = min(round(0.006 * raw_cycles), n_fail)
    if pilot_fail:
        chosen = np.argsort(propensity[raw_positions])[
            -pilot_fail:
        ]
        failed[raw_positions[chosen]] = 1
    if n_fail - pilot_fail:
        candidates = np.setdiff1d(idx, raw_positions)
        remaining = np.argsort(propensity[candidates])[
            -(n_fail - pilot_fail) :
        ]
        failed[candidates[remaining]] = 1
    starts = pd.Timestamp("2025-09-01") + pd.to_timedelta(
        source_idx * 32, unit="s"
    )
    meta = pd.DataFrame(
        {
            "serial": source_idx + 1,
            "cycle_start": starts,
            "failed": failed,
            "machine_id": [
                f"TUL-CNC-{i % 14 + 1:02d}" for i in source_idx
            ],
        }
    )
    meta["inspection_completed_at"] = (
        meta.cycle_start + pd.Timedelta(hours=4)
    )
    prior = np.zeros(cycles, dtype=int)
    # Count the last five same-cell inspection results available
    # strictly before each cycle start, including missing joins.
    for machine in range(14):
        positions = idx[source_idx % 14 == machine]
        ticks = source_idx[positions]
        available = (
            np.searchsorted(ticks, ticks - 450, side="left") - 1
        )
        for lag in range(5):
            old = available - lag
            valid = old >= 0
            prior[positions[valid]] += failed[
                positions[old[valid]]
            ]
    context = pd.DataFrame(
        {
            "nominal_seconds": 48.0,
            "tool_life_pct": 100 * (1 - wear),
            "prior5_fails": prior,
            "part_family": np.resize(
                ["fitting", "rib-web", "bracket"], cycles
            ),
            "machine_id": meta.machine_id,
            "shift": ((starts.hour // 8) + 1).astype(str),
        }
    )
    if raw_cycles:
        raw = np.lib.format.open_memmap(
            output / "windows.npy",
            mode="w+",
            dtype="float32",
            shape=(raw_cycles, 5, 512),
        )
        raw[:] = 0
        np.save(output / "labels.npy", failed[raw_positions])
        np.save(
            output / "valid_lengths.npy",
            np.full(raw_cycles, 480, dtype=np.int16),
        )
    writer = None
    t = np.arange(480) / FS
    try:
        for start in range(0, cycles, batch_size):
            stop = min(start + batch_size, cycles)
            # RNG consumption does not depend on batch size.
            noise = rng.normal(size=(stop - start, 5, 480))
            w = wear[start:stop, None]
            windows = np.empty_like(noise)
            windows[:, 0] = (
                4800
                + 30 * noise[:, 0]
                + 20 * np.sin(2 * np.pi * 0.3 * t)
            )
            windows[:, 1] = 240 + 3 * noise[:, 1]
            windows[:, 2] = (0.4 + 0.3 * w) * noise[
                :, 2
            ] + 0.5 * w * np.sin(2 * np.pi * 3 * t)
            windows[:, 3] = 140 + 35 * w + 2 * noise[:, 3]
            windows[:, 4] = 12 - 0.5 * w + 0.1 * noise[:, 4]
            windows = windows.astype(np.float32)
            feature = batch_features(
                windows, context.iloc[start:stop]
            )
            assert feature.shape[1] == 51
            for col in [
                "serial",
                "cycle_start",
                "failed",
                "inspection_completed_at",
            ]:
                feature[col] = (
                    meta[col].iloc[start:stop].to_numpy()
                )
            table = pa.Table.from_pandas(
                feature, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(
                    output / "cycle_features.parquet",
                    table.schema,
                )
            writer.write_table(table)
            selected = (raw_positions >= start) & (
                raw_positions < stop
            )
            if selected.any():
                raw[np.flatnonzero(selected), :, :480] = windows[
                    raw_positions[selected] - start
                ]
        if raw_cycles:
            raw.flush()
            del raw
    finally:
        if writer is not None:
            writer.close()
    meta.to_parquet(output / "cycles.parquet", index=False)
    meta.iloc[raw_positions].to_parquet(
        output / "pilot_cycles.parquet", index=False
    )
    raw_idx = np.arange(raw_count)
    raw_meta = pd.DataFrame(
        {
            "serial": raw_idx + 1,
            "cycle_start": pd.Timestamp("2025-09-01")
            + pd.to_timedelta(raw_idx * 32, unit="s"),
            "machine_id": [
                f"TUL-CNC-{i % 14 + 1:02d}" for i in raw_idx
            ],
        }
    )
    raw_meta.to_parquet(
        output / "cnc_cycles.parquet", index=False
    )
    meta[
        ["serial", "failed", "inspection_completed_at"]
    ].to_parquet(output / "cnc_inspections.parquet", index=False)
    manifest = {
        "synthetic": True,
        "generator_version": 1,
        "seed": 20260914,
        "cycles": cycles,
        "raw_cycles": raw_count,
        "unmatched_cycles": raw_count - cycles,
        "failures": int(failed.sum()),
        "raw_windows": raw_cycles,
        "raw_failures": int(failed[raw_positions].sum()),
        "feature_count": 51,
        "sample_rate_hz": 10,
        "samples_per_cycle": 480,
        "padded_samples": 512,
        "note": "Fourteen-cell simulation; the raw pilot is TUL-CNC-07. "
        "Family durations are equal in this fixture. "
        "It does not reproduce the illustrative performance tables.",
        "files": {},
    }
    for path in sorted(output.iterdir()):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(
                lambda: stream.read(1024 * 1024), b""
            ):
                digest.update(block)
        manifest["files"][path.name] = digest.hexdigest()
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/cnc")
    )
    parser.add_argument("--cycles", type=int, default=931240)
    parser.add_argument("--raw-cycles", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    print(
        json.dumps(
            generate(
                args.output,
                args.cycles,
                args.raw_cycles,
                args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
