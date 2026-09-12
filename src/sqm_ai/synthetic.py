"""Deterministic, fictional teaching data; never production evidence.

Run ``python -m sqm_ai.synthetic --output data`` from the repo.
The generator, its contracts, and its manifest are versioned;
generated data are ignored by Git. See docs/SYNTHETIC_DATA.md.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260911
COMPLETE_THROUGH = pd.Timestamp("2026-09-01")
MONTHS = pd.date_range("2023-09-01", periods=36, freq="MS")
CATEGORIES = ["dimensional", "cosmetic", "material", "functional"]
SPECIAL = {
    "S-0417": "Cobalt Machining",
    "S-1130": "Redline Castings",
    "S-0088": "Apex Fastener Group",
}


def suppliers() -> pd.DataFrame:
    ids = list(SPECIAL) + [
        f"S-{i:04d}"
        for i in range(1, 1801)
        if f"S-{i:04d}" not in SPECIAL
    ]
    df = pd.DataFrame({"supplier_id": ids})
    df["supplier_name"] = [
        SPECIAL.get(s, f"Synthetic Works {s}") for s in ids
    ]
    df["tier"] = np.resize(["A", "B", "C"], len(df))
    df["active"] = True
    df["onboarded_at"] = pd.Timestamp("2015-09-01")
    # 126*18 + 14 = 2,282 pre-onboarding labeled rows excluded.
    df.loc[df.index[-126:], "onboarded_at"] = MONTHS[18]
    df.loc[df.index[-127], "onboarded_at"] = MONTHS[14]
    return df


def make_ncrs(s: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    # Exact Pareto crossing at 214 and 1,106 suppliers with >=5.
    counts = [612, 498, 455] + [105] * 21 + [104] * 190
    counts += [6] * 728 + [5] * 164 + [1] * 694
    records = []
    for i, row in s.iterrows():
        for year in range(3):
            months = MONTHS[year * 12 : (year + 1) * 12]
            months = months[months >= row.onboarded_at]
            if not len(months):
                continue
            n = counts[i]
            allocations = np.full(len(months), n // len(months))
            allocations[: n % len(months)] += 1
            if year == 2 and i == 0:
                allocations = [53] * 5 + [52] * 4 + [46, 46, 47]
            elif year == 2 and i == 1:
                allocations = [42] * 8 + [41] * 3 + [39]
            elif year == 2 and i == 2:
                allocations = [40] * 8 + [41] * 3 + [12]
            for month, count in zip(months, allocations):
                for k in range(count):
                    discovered = month + pd.Timedelta(
                        days=int(rng.integers(1, 27)),
                        hours=int(rng.integers(0, 24)),
                    )
                    records.append((row.supplier_id, discovered))
    n = pd.DataFrame(
        records, columns=["supplier_id", "discovered_at"]
    )
    n = n.sort_values(["discovered_at", "supplier_id"])
    n = n.reset_index(drop=True)
    years = n.discovered_at.dt.year
    serial = n.groupby(years).cumcount() + 1
    n["ncr_id"] = [
        f"NCR-{y}-{k:04d}" for y, k in zip(years, serial)
    ]
    n["category"] = rng.choice(CATEGORIES, len(n))
    recent = n.discovered_at >= "2025-09-01"
    category_counts = [13535, 7062, 6185, 2630]
    category = np.repeat(CATEGORIES, category_counts)
    rng.shuffle(category)
    n.loc[recent, "category"] = category
    n["severity"] = rng.integers(1, 3, len(n))
    # The three familiar suppliers retain their printed means.
    for sid, total in [
        ("S-0417", 1469),
        ("S-1130", 1444),
        ("S-0088", 819),
    ]:
        idx = n.index[recent & n.supplier_id.eq(sid)].to_numpy()
        vals = np.full(len(idx), total // len(idx))
        vals[: total % len(idx)] += 1
        rng.shuffle(vals)
        if sid == "S-0417":
            aug = n.loc[idx, "discovered_at"].ge("2026-08-01")
            vals[aug] = [5] * 6 + [4] * 3 + [2] * 38
            prior = np.flatnonzero(~aug)
            remainder = total - int(vals[aug].sum())
            vals[prior] = remainder // len(prior)
            vals[prior[: remainder % len(prior)]] += 1
        n.loc[idx, "severity"] = vals
    # Add persistent episodes until the complete-window label
    # count is 5,483 / 57,118 (rounds to the printed 0.096).
    label_keys = [
        (row.supplier_id, m)
        for row in s.itertuples()
        for m in MONTHS[:33]
        if m >= row.onboarded_at
    ]
    key_index = {key: i for i, key in enumerate(label_keys)}
    covered = np.zeros(len(label_keys), dtype=bool)
    affected = []
    for row in n.itertuples():
        matches = []
        for lag in range(1, 5):
            m = row.discovered_at.to_period("M").to_timestamp()
            m -= pd.offsets.MonthBegin(lag)
            cutoff = m + pd.offsets.MonthBegin(1)
            key = (row.supplier_id, m)
            if (
                key in key_index
                and cutoff
                <= row.discovered_at
                < cutoff + pd.Timedelta(days=90)
            ):
                matches.append(key_index[key])
        affected.append(matches)
        if row.severity >= 3:
            covered[matches] = True
    remaining = 5483 - int(covered.sum())
    # Gumbel priorities favor recurrent high-volume suppliers,
    # while allowing rare suppliers to have positive outcomes.
    rank = {sid: i for i, sid in enumerate(s.supplier_id)}
    weight = np.array(
        [1 / (1 + rank[x] / 100) for x in n.supplier_id]
    )
    priority = np.log(weight) + rng.gumbel(size=len(n))
    for j in np.argsort(-priority):
        if n.at[j, "supplier_id"] in SPECIAL:
            continue
        positions = affected[j]
        added = int((~covered[positions]).sum())
        if 0 < added <= remaining:
            n.at[j, "severity"] = int(
                rng.choice([3, 4, 5], p=[0.68, 0.27, 0.05])
            )
            covered[positions] = True
            remaining -= added
        if remaining == 0:
            break
    if remaining:
        raise RuntimeError("could not satisfy label contract")
    n["part_number"] = rng.choice(
        ["7741-B", "NL-2214", "FAST-08", "CAST-17"], len(n)
    )
    n["quantity"] = rng.integers(1, 31, len(n))
    n["disposition"] = np.where(
        n.severity >= 4, "scrap", "rework"
    )
    n["cost_impact_usd"] = np.round(
        rng.uniform(50, 9000, len(n)), 2
    )
    cobalt = recent & n.supplier_id.eq("S-0417")
    n.loc[n.index[cobalt][0], "cost_impact_usd"] = 48200.0
    missing = rng.choice(
        n.index[recent & ~cobalt], 4806, replace=False
    )
    n.loc[missing, "cost_impact_usd"] = np.nan
    # Closed statuses are observable by the snapshot watermark.
    n["closed_at"] = n.discovered_at + pd.to_timedelta(
        rng.integers(1, 31, len(n)), unit="D"
    )
    n["closed_at"] = n.closed_at.clip(
        upper=COMPLETE_THROUGH - pd.Timedelta(seconds=1)
    )
    n.loc[
        rng.choice(n.index[recent], 2317, replace=False),
        "closed_at",
    ] = pd.NaT
    n["owner"] = "Synthetic Quality Engineer"
    n["description"] = [
        f"Synthetic training example: {c} nonconformance on {p}."
        for c, p in zip(n.category, n.part_number)
    ]
    # Demo availability is immediate; real systems need recorded_at.
    n["recorded_at"] = n.discovered_at
    return n


def make_supplier_month(
    s: pd.DataFrame, n: pd.DataFrame
) -> pd.DataFrame:
    """Snapshot features use events before each exclusive cutoff."""
    rng = np.random.default_rng(SEED + 1)
    event_groups = {sid: g for sid, g in n.groupby("supplier_id")}
    rows = []
    for supplier in s.itertuples():
        events = event_groups.get(supplier.supplier_id)
        observed_months = MONTHS[MONTHS >= supplier.onboarded_at]
        # Each month has 100 fictional complete PO-line receipts,
        # all on day 15. This makes the delivery denominators explicit.
        received_units = 100 * rng.integers(
            15, 51, len(observed_months)
        )
        receipt_dates = observed_months + pd.Timedelta(days=14)
        on_time = rng.binomial(100, 0.92, len(observed_months))
        days_late = rng.uniform(1, 7, len(observed_months))
        for sid, last_on_time in [
            ("S-0417", 83),
            ("S-1130", 88),
            ("S-0088", 96),
        ]:
            if supplier.supplier_id == sid:
                on_time[-1] = last_on_time
        days_late *= 1 - on_time / 100
        for m in MONTHS:
            if m < supplier.onboarded_at:
                continue
            cutoff = m + pd.offsets.MonthBegin(1)
            current = events[
                events.discovered_at.ge(m)
                & events.discovered_at.lt(cutoff)
            ]
            trailing = events[
                events.discovered_at.ge(
                    cutoff - pd.Timedelta(days=90)
                )
                & events.discovered_at.lt(cutoff)
            ]
            future = events[
                events.discovered_at.ge(cutoff)
                & events.discovered_at.lt(
                    cutoff + pd.Timedelta(days=90)
                )
            ]
            mature = (
                cutoff + pd.Timedelta(days=90) <= COMPLETE_THROUGH
            )
            q = min(len(trailing) / 150, 1.0)
            fpy = float(
                np.clip(
                    0.978 - 0.04 * q + rng.normal(0, 0.012),
                    0.7,
                    1,
                )
            )
            audit = float(
                np.clip(94 - 15 * q + rng.normal(0, 5), 45, 100)
            )
            month_i = observed_months.get_loc(m)
            received = int(received_units[month_i])
            deliveries = (
                receipt_dates >= cutoff - pd.Timedelta(days=90)
            ) & (receipt_dates < cutoff)
            units_90d = int(received_units[deliveries].sum())
            row = {
                "supplier_id": supplier.supplier_id,
                "tier": supplier.tier,
                "month": m,
                "onboarded_at": supplier.onboarded_at,
                "ncr_count": len(current),
                "sev3_plus_count": int(
                    (current.severity >= 3).sum()
                ),
                "avg_severity": current.severity.mean(),
                "cost_impact_usd": current.cost_impact_usd.sum(
                    min_count=len(current)
                )
                if len(current)
                else 0.0,
                "fpy": fpy,
                "otd": float(on_time[month_i] / 100),
                "audit_score": audit,
                "open_cars": int(rng.integers(0, 5)),
                "units_received": received,
                "units_received_90d": units_90d,
                "spend_usd": received * 25.0,
                "spend_90d": units_90d * 25.0,
                "otd_pct": float(
                    on_time[deliveries].mean() / 100
                ),
                "receipt_count": 100,
                "on_time_receipt_count": int(on_time[month_i]),
                "receipt_date": receipt_dates[month_i],
                "days_late": float(days_late[month_i]),
                "ncr_count_90d": len(trailing),
                "sev3_count_90d": int(
                    (trailing.severity >= 3).sum()
                ),
                "avg_severity_90d": trailing.severity.mean(),
                "avg_days_late": float(
                    days_late[deliveries].mean()
                ),
                "audit_score_last": audit,
                "audit_date_last": max(
                    supplier.onboarded_at,
                    cutoff - pd.Timedelta(days=30),
                ),
                "car_response_days": max(
                    1.0, rng.normal(12 + q * 10, 4)
                ),
                "car_effectiveness": float(rng.uniform(0.5, 1))
                if month_i >= 6
                else np.nan,
                "sev3_next_90d": int((future.severity >= 3).any())
                if mature
                else None,
                "harm_next_90d": int(future.severity.sum())
                if mature
                else None,
                "label_end": cutoff + pd.Timedelta(days=90),
                "data_complete_through": COMPLETE_THROUGH,
            }
            row.update(
                {
                    c + "_count": int(
                        current.category.eq(c).sum()
                    )
                    for c in CATEGORIES
                }
            )
            rows.append(row)
    df = pd.DataFrame(rows)
    for sid, fpy, otd in [
        ("S-0417", 0.912, 0.83),
        ("S-1130", 0.933, 0.88),
        ("S-0088", 0.981, 0.96),
    ]:
        mask = df.supplier_id.eq(sid) & df.month.eq(MONTHS[-1])
        df.loc[mask, ["fpy", "otd"]] = [fpy, otd]
    mask = df.supplier_id.eq("S-0417") & df.month.eq(MONTHS[-1])
    df.loc[
        mask, ["audit_score", "audit_score_last", "open_cars"]
    ] = [71.0, 71.0, 4]
    g = df.groupby("supplier_id", sort=False)
    df["ncr_count_3mo"] = g.ncr_count.transform(
        lambda x: x.rolling(3, min_periods=1).sum()
    )
    # Missing records are simulated independently of future labels.
    for col, rate in [
        ("fpy", 0.02),
        ("audit_score_last", 0.03),
        ("car_response_days", 0.08),
    ]:
        missing = (
            rng.random(len(df)) < rate
        ) & ~df.supplier_id.isin(SPECIAL)
        df.loc[missing, col] = np.nan
    df["audit_score"] = df.audit_score_last
    from sqm_ai.build1.features import add_derived

    df = add_derived(df)
    df["limited_history"] = (
        df.groupby("supplier_id").cumcount() < 5
    )
    df["sev3_next_90d"] = df.sev3_next_90d.astype("Int64")
    return df


def cobalt_lots() -> pd.DataFrame:
    # Moment-matched teaching fixture, not recovered real measurements.
    rng = np.random.default_rng(774)
    rows = []
    for period, count, mean, sd in [
        ("before", 41, 0.9502, 0.0271),
        ("after", 38, 0.9202, 0.0378),
    ]:
        x = rng.beta(5, 2, count)
        x = mean + sd * (x - x.mean()) / x.std(ddof=1)
        if period == "before":
            x = np.random.default_rng(205).permutation(x)
        rows.extend(
            {"period": period, "fpy": float(v)} for v in x
        )
    return pd.DataFrame(rows)


def generate(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest_file = output / "synthetic_manifest.json"
    expected = [
        "suppliers.parquet",
        "ncrs.parquet",
        "ncrs_2025-09_2026-08.csv",
        "supplier_month.parquet",
        "supplier_month_features.parquet",
        "supplier_month_all.parquet",
        "cobalt_lot_fpy.csv",
        "synthetic_manifest.json",
    ]
    if any((output / name).exists() for name in expected):
        raise FileExistsError(
            "output already contains generated names; use a new directory"
        )
    s = suppliers()
    n = make_ncrs(s)
    sm = make_supplier_month(s, n)
    labeled = sm.dropna(subset=["sev3_next_90d"]).copy()
    labeled["sev3_next_90d"] = labeled.sev3_next_90d.astype(
        "int64"
    )
    export = n[n.discovered_at.ge("2025-09-01")].merge(
        s[["supplier_id", "supplier_name"]], on="supplier_id"
    )
    columns = [
        "ncr_id",
        "supplier_id",
        "supplier_name",
        "part_number",
        "category",
        "severity",
        "quantity",
        "disposition",
        "cost_impact_usd",
        "discovered_at",
        "closed_at",
    ]
    export = export[columns]
    counts = export.supplier_id.value_counts()
    assert export.shape == (29412, 11)
    assert (
        int(counts.cumsum().searchsorted(0.8 * len(export))) + 1
        == 214
    )
    assert int((counts >= 5).sum()) == 1106
    assert len(labeled) == 57118
    assert int(labeled.sev3_next_90d.sum()) == 5483
    assert export.ncr_id.is_unique
    s.to_parquet(output / "suppliers.parquet", index=False)
    n.to_parquet(output / "ncrs.parquet", index=False)
    export.to_csv(
        output / "ncrs_2025-09_2026-08.csv", index=False
    )
    sm.to_parquet(
        output / "supplier_month_all.parquet", index=False
    )
    labeled.to_parquet(
        output / "supplier_month.parquet", index=False
    )
    labeled.to_parquet(
        output / "supplier_month_features.parquet", index=False
    )
    cobalt_lots().to_csv(
        output / "cobalt_lot_fpy.csv", index=False
    )
    manifest = {
        "generator_version": 1,
        "seed": SEED,
        "synthetic": True,
        "data_complete_through": "2026-09-01",
        "supplier_month_rows": len(sm),
        "labeled_rows": len(labeled),
        "positives": int(labeled.sev3_next_90d.sum()),
        "ncr_export_rows": len(export),
        "files": {},
    }
    for name in expected[:-1]:
        manifest["files"][name] = hashlib.sha256(
            (output / name).read_bytes()
        ).hexdigest()
    manifest_file.write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data")
    )
    args = parser.parse_args()
    print(json.dumps(generate(args.output), indent=2))


if __name__ == "__main__":
    main()
