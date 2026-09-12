"""Verify synthetic data independently of generator assertions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats

from sqm_ai.stats import bootstrap_ci, bootstrap_diff_ci


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_hashes(folder: Path, manifest_name: str):
    manifest = json.loads((folder / manifest_name).read_text())
    require(
        manifest["synthetic"] is True,
        "missing synthetic provenance",
    )
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256()
        with (folder / name).open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        require(
            digest.hexdigest() == expected,
            f"hash mismatch: {name}",
        )
    return manifest


def verify_supplier(folder: Path):
    manifest = check_hashes(folder, "synthetic_manifest.json")
    n = pd.read_parquet(folder / "ncrs.parquet")
    sm = pd.read_parquet(folder / "supplier_month_all.parquet")
    labeled = pd.read_parquet(folder / "supplier_month.parquet")
    export = pd.read_csv(folder / "ncrs_2025-09_2026-08.csv")
    require(export.shape == (29412, 11), "NCR shape")
    require(export.ncr_id.is_unique, "duplicate NCR IDs")
    require(
        int(export.cost_impact_usd.isna().sum()) == 4806,
        "missing costs",
    )
    require(
        int(export.closed_at.isna().sum()) == 2317, "open NCRs"
    )
    counts = export.supplier_id.value_counts()
    require(
        counts.iloc[:3].to_dict()
        == {"S-0417": 612, "S-1130": 498, "S-0088": 455},
        "top suppliers",
    )
    require(len(counts) == 1800, "active supplier coverage")
    require(
        int(counts.cumsum().searchsorted(0.8 * len(export))) + 1
        == 214,
        "Pareto crossing",
    )
    require(
        int((counts >= 5).sum()) == 1106, "NCR count threshold"
    )
    require(
        export.category.value_counts().to_dict()
        == {
            "dimensional": 13535,
            "cosmetic": 7062,
            "material": 6185,
            "functional": 2630,
        },
        "category counts",
    )
    require(
        len(labeled) == 57118 and sm.month.nunique() == 36,
        "supplier month coverage",
    )
    require(labeled.sev3_next_90d.sum() == 5483, "positive count")
    require(
        int(sm.month.ge("2025-09-01").sum()) == 21600,
        "Chapter 4 twelve-month slice",
    )
    require(
        not sm.duplicated(["supplier_id", "month"]).any(),
        "duplicate panel rows",
    )
    require(
        (
            sm.month
            >= sm.onboarded_at.dt.to_period("M").dt.to_timestamp()
        ).all(),
        "pre-onboarding rows",
    )
    complete = pd.Timestamp(manifest["data_complete_through"])
    require(
        sm.loc[sm.label_end > complete, "sev3_next_90d"]
        .isna()
        .all(),
        "right-censored labels incorrectly assigned",
    )
    # Recalculate EVERY forward target and trailing count by a
    # separate sorted cumulative-sum algorithm (not generator filters).
    for sid, rows in sm.groupby("supplier_id"):
        events = n[n.supplier_id.eq(sid)].sort_values(
            "discovered_at"
        )
        times = events.discovered_at.to_numpy(
            dtype="datetime64[ns]"
        )
        cutoffs = rows.month + pd.offsets.MonthBegin(1)
        cuts = cutoffs.to_numpy(dtype="datetime64[ns]")
        ends = (cutoffs + pd.Timedelta(days=90)).to_numpy(
            dtype="datetime64[ns]"
        )
        starts = (cutoffs - pd.Timedelta(days=90)).to_numpy(
            dtype="datetime64[ns]"
        )
        left, right = (
            np.searchsorted(times, cuts),
            np.searchsorted(times, ends),
        )
        trailing = left - np.searchsorted(times, starts)
        require(
            np.array_equal(trailing, rows.ncr_count_90d),
            "trailing NCR count",
        )
        positive_sum = np.r_[
            0, (events.severity.to_numpy() >= 3).cumsum()
        ]
        harm_sum = np.r_[0, events.severity.to_numpy().cumsum()]
        mature = rows.label_end.le(complete).to_numpy()
        require(
            np.array_equal(
                (positive_sum[right] - positive_sum[left] > 0)[
                    mature
                ],
                rows.sev3_next_90d[mature].astype(int),
            ),
            "future label mismatch",
        )
        require(
            np.array_equal(
                (harm_sum[right] - harm_sum[left])[mature],
                rows.harm_next_90d[mature],
            ),
            "future harm mismatch",
        )
    cobalt = sm[
        sm.supplier_id.eq("S-0417") & sm.month.eq("2026-08-01")
    ].iloc[0]
    require(
        (
            cobalt.ncr_count,
            cobalt.sev3_plus_count,
            cobalt.ncr_count_3mo,
        )
        == (47, 9, 139),
        "Cobalt monthly counts",
    )
    require(
        (
            cobalt.fpy,
            cobalt.otd,
            cobalt.audit_score,
            cobalt.open_cars,
        )
        == (0.912, 0.83, 71, 4),
        "Cobalt monthly metrics",
    )
    lots = pd.read_csv(folder / "cobalt_lot_fpy.csv")
    a = lots.loc[lots.period.eq("before"), "fpy"].to_numpy()
    b = lots.loc[lots.period.eq("after"), "fpy"].to_numpy()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    diff = tuple(np.round(bootstrap_diff_ci(a, b, np.mean), 4))
    mean_ci = tuple(np.round(bootstrap_ci(b, np.mean), 4))
    require(
        (round(t, 2), round(p, 5)) == (4.03, 0.00015),
        "Welch example",
    )
    require(
        round(stats.mannwhitneyu(a, b).pvalue, 5) == 0.00047,
        "rank test example",
    )
    require(
        diff == (-0.0446, -0.0161)
        and mean_ci == (0.9083, 0.9320),
        "bootstrap example",
    )
    return {
        "status": "passed",
        "ncr_rows": len(export),
        "panel_rows": len(sm),
        "mature_rows": len(labeled),
        "positives": int(labeled.sev3_next_90d.sum()),
        "cobalt_difference_ci": diff,
        "cobalt_mean_ci": mean_ci,
    }


def verify_cnc(folder: Path):
    m = check_hashes(folder, "manifest.json")
    file = pq.ParquetFile(folder / "cycle_features.parquet")
    require(
        file.metadata.num_rows == m["cycles"], "CNC row count"
    )
    total = 0
    for batch in file.iter_batches(batch_size=10000):
        df = batch.to_pandas()
        require(
            df.select_dtypes("number").notna().all().all(),
            "missing CNC features",
        )
        total += int(df.failed.sum())
    require(total == m["failures"], "CNC failure count")
    raw = pd.read_parquet(folder / "cnc_cycles.parquet")
    inspections = pd.read_parquet(
        folder / "cnc_inspections.parquet"
    )
    require(
        raw.serial.is_unique and inspections.serial.is_unique,
        "duplicate cycle serial",
    )
    joined = raw.merge(
        inspections,
        on="serial",
        how="left",
        validate="one_to_one",
    )
    require(
        int(joined.failed.isna().sum()) == m["unmatched_cycles"],
        "unmatched cycles",
    )
    if m["raw_windows"]:
        windows = np.load(folder / "windows.npy", mmap_mode="r")
        labels = np.load(folder / "labels.npy")
        require(
            windows.shape == (m["raw_windows"], 5, 512),
            "window shape",
        )
        require(
            int(labels.sum()) == m["raw_failures"],
            "pilot failures",
        )
        pilot = pd.read_parquet(folder / "pilot_cycles.parquet")
        require(
            pilot.machine_id.eq("TUL-CNC-07").all(), "pilot cell"
        )
        require(
            np.array_equal(labels, pilot.failed),
            "pilot alignment",
        )
    return {
        "status": "passed",
        "cycles": m["cycles"],
        "failures": total,
        "raw_cycles": m["raw_cycles"],
        "unmatched": m["unmatched_cycles"],
        "features": m["feature_count"],
        "pilot_windows": m["raw_windows"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--cnc", type=Path)
    args = parser.parse_args()
    result = {"supplier": verify_supplier(args.data)}
    if args.cnc:
        result["cnc"] = verify_cnc(args.cnc)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
