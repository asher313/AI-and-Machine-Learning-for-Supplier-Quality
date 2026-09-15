"""Decision-policy evaluation and a separately reserved final-period rehearsal."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def monthly_policy_report(predictions, k=50):
    """Compare harm ranking with recent burden on the same active monthly cohort.

    Limited-history flags remain visible; they do not exclude audit candidates.
    This is audit planning, not the separate probability escalation policy.
    """
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    required = [
        "month",
        "supplier_id",
        "predicted_harm_90d",
        "recent_harm_90d",
        "sev3_next_90d",
        "harm_next_90d",
    ]
    df = predictions[required].copy()
    if df.empty or df.isna().any().any():
        raise ValueError(
            "complete nonempty policy evaluation rows required"
        )
    df["month"] = pd.to_datetime(df.month, errors="raise")
    if df.duplicated(["month", "supplier_id"]).any():
        raise ValueError("duplicate monthly candidate")
    if not df.supplier_id.map(
        lambda x: isinstance(x, str) and bool(x)
    ).all():
        raise ValueError("nonempty string supplier IDs required")
    values = df[required[2:]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(
            "finite nonnegative predictions and outcomes required"
        )
    if not df.sev3_next_90d.isin([0, 1]).all():
        raise ValueError("binary serious-event outcome required")
    rows = []
    for month, cohort in df.groupby("month", sort=True):
        total_positive = int(cohort.sev3_next_90d.sum())
        total_harm = float(cohort.harm_next_90d.sum())
        for policy, column in [
            ("predicted_harm", "predicted_harm_90d"),
            ("recent_harm_baseline", "recent_harm_90d"),
        ]:
            chosen = cohort.sort_values(
                [column, "supplier_id"], ascending=[False, True]
            ).head(k)
            positive = int(chosen.sev3_next_90d.sum())
            harm = float(chosen.harm_next_90d.sum())
            rows.append(
                {
                    "month": str(month.date()),
                    "policy": policy,
                    "eligible_suppliers": len(cohort),
                    "requested_k": k,
                    "selected_suppliers": len(chosen),
                    "selected_ids": chosen.supplier_id.tolist(),
                    "serious_events_selected": positive,
                    "serious_events_total": total_positive,
                    "precision_at_k": positive / len(chosen),
                    "recall_at_k": positive / total_positive
                    if total_positive
                    else None,
                    "harm_selected": harm,
                    "harm_total": total_harm,
                    "captured_harm_fraction": harm / total_harm
                    if total_harm
                    else None,
                }
            )
    return {
        "objective": "Capture subsequent ordinal severity burden within monthly audit capacity",
        "tie_break": "ascending supplier_id",
        "rows": rows,
        "interpretation": "Descriptive cohort results; not causal audit benefit or production approval",
    }


def prediction_frame(df, harm):
    raw = np.asarray(harm, dtype=float)
    if len(raw) != len(df) or not np.isfinite(raw).all():
        raise ValueError(
            "one finite prediction per candidate required"
        )
    result = df[
        ["month", "supplier_id", "sev3_next_90d", "harm_next_90d"]
    ].copy()
    result["predicted_harm_90d"] = np.maximum(0, raw)
    # Zero NCRs imply zero observed burden even if their mean severity is absent.
    recent = df.ncr_count_90d * df.avg_severity_90d
    result["recent_harm_90d"] = recent.mask(
        df.ncr_count_90d.eq(0), 0
    )
    return result


def reserve_period(df, final_start):
    """Separate whole monthly cohorts and purge not-yet-mature training labels."""
    from sqm_ai.build1.features import load_modelling_frame

    # Enforce the documented 90-day target horizon before trusting dates for
    # the purge. An accidentally shortened label_end must not admit a row
    # whose real forward outcome was unavailable at the prediction cutoff.
    df = df.copy()
    for column in ("month", "label_end", "data_complete_through"):
        df[column] = pd.to_datetime(df[column], errors="raise")
        if df[column].dt.tz is not None:
            raise ValueError("timezone-naive row dates required")
    if not df.month.eq(
        df.month.dt.to_period("M").dt.start_time
    ).all():
        raise ValueError(
            "row month must be a complete month-start identity"
        )
    expected_end = (
        df.month
        + pd.offsets.MonthBegin(1)
        + pd.Timedelta(days=90)
    )
    if not df.label_end.eq(expected_end).all():
        raise ValueError(
            "label_end must equal the next month start plus 90 days"
        )
    load_modelling_frame(df)
    start = pd.Timestamp(final_start)
    if (
        pd.isna(start)
        or start.tzinfo is not None
        or start != start.to_period("M").start_time
    ):
        raise ValueError(
            "final_start must be a timezone-naive month start"
        )
    if (
        df[["supplier_id", "month", "label_end"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "complete row identities and label dates required"
        )
    if df.duplicated(["supplier_id", "month"]).any():
        raise ValueError("duplicate supplier-month")
    cutoff = start + pd.offsets.MonthBegin(1)
    development = df[
        df.month.lt(start) & df.label_end.le(cutoff)
    ].copy()
    final = df[df.month.ge(start)].copy()
    if development.empty or final.empty:
        raise ValueError(
            "nonempty development and final periods required"
        )
    return (
        development,
        final,
        {
            "final_first_month": str(start.date()),
            "first_prediction_cutoff": str(cutoff.date()),
            "development_rows": len(development),
            "final_rows": len(final),
            "purged_rows": len(df)
            - len(development)
            - len(final),
            "development_max_label_end": str(
                development.label_end.max().date()
            ),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    reserve = sub.add_parser("reserve")
    reserve.add_argument("--input", type=Path, required=True)
    reserve.add_argument("--final-start", required=True)
    reserve.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate-final")
    evaluate.add_argument("--partition", type=Path, required=True)
    evaluate.add_argument("--models", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(
            "use a new output path; retain the prior evaluation"
        )
    if args.command == "reserve":
        df = pd.read_parquet(args.input)
        development, final, manifest = reserve_period(
            df, args.final_start
        )
        args.output.mkdir(parents=True)
        development.to_parquet(
            args.output / "development.parquet", index=False
        )
        final.to_parquet(
            args.output / "final.parquet", index=False
        )
        manifest.update(
            {
                "source_sha256": digest(args.input),
                "development_sha256": digest(
                    args.output / "development.parquet"
                ),
                "final_sha256": digest(
                    args.output / "final.parquet"
                ),
                "status": "Partition rehearsal; isolation alone cannot prove no human or earlier experiment used these outcomes",
            }
        )
        (args.output / "split.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        print(json.dumps(manifest, indent=2))
        return

    from sklearn.metrics import (
        brier_score_loss,
        recall_score,
        roc_auc_score,
    )

    from sqm_ai.build1.features import load_modelling_frame
    from sqm_ai.build1.score_suppliers import load_artifacts

    manifest = json.loads(
        (args.partition / "split.json").read_text()
    )
    for name in ("development", "final"):
        if (
            digest(args.partition / f"{name}.parquet")
            != manifest[f"{name}_sha256"]
        ):
            raise ValueError("partition checksum mismatch")
    development = pd.read_parquet(
        args.partition / "development.parquet"
    )
    final = pd.read_parquet(args.partition / "final.parquet")
    checked_development, checked_final, checked = reserve_period(
        pd.concat([development, final], ignore_index=True),
        manifest["final_first_month"],
    )
    # Counts alone miss a swapped development/final row. concat(ignore_index)
    # assigns each source file a disjoint positional range; verify membership.
    if (
        checked["development_rows"] != len(development)
        or checked["final_rows"] != len(final)
        or not (
            checked_development.index < len(development)
        ).all()
        or not (checked_final.index >= len(development)).all()
    ):
        raise ValueError(
            "partition violates time/label-maturity separation"
        )
    reg, clf, cols, threshold, metadata = load_artifacts(
        args.models
    )
    if (
        metadata["training_data_sha256"]
        != manifest["development_sha256"]
    ):
        raise ValueError(
            "model was not fitted to this reserved development dataset"
        )
    X, y, _ = load_modelling_frame(final)
    if list(X.columns) != cols or y.nunique() != 2:
        raise ValueError(
            "matching feature contract and both final outcome classes required"
        )
    p = clf.predict_proba(X)[:, 1]
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid final probabilities")
    predictions = prediction_frame(final, reg.predict(X))
    report = monthly_policy_report(predictions)
    report.update(
        {
            "evaluation_kind": "reserved-period synthetic protocol rehearsal",
            "model_version": metadata["model_version"],
            "artifact_sha256": metadata["artifact_sha256"],
            "split_manifest_sha256": digest(
                args.partition / "split.json"
            ),
            "n_test": len(final),
            "positive_test": int(y.sum()),
            "roc_auc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "screen_threshold": 0.04,
            "screen_recall": float(recall_score(y, p >= 0.04)),
            "illustrative_escalation_threshold": threshold,
            "release_approved": False,
            "limitation": "Prespecify selection and retain a truly untouched period for real validation. Published synthetic outcomes are not newly independent evidence.",
        }
    )
    args.output.mkdir(parents=True)
    predictions.to_parquet(
        args.output / "predictions.parquet", index=False
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "model_version",
                    "n_test",
                    "roc_auc",
                    "release_approved",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
