from datetime import date

import pytest
from sqlalchemy import text

from sqm_ai.build1.labels import build_labels
from sqm_ai.features import build_supplier_month

pytestmark = pytest.mark.integration


def test_incomplete_windows_are_unknown_not_negative(db):
    build_supplier_month(
        db.url.render_as_string(hide_password=False)
    )
    labels = build_labels(db, date(2026, 9, 1))
    assert (
        labels.loc[labels.month >= "2026-06-01", "sev3_next_90d"]
        .isna()
        .all()
    )
    may = labels[
        (labels.month == "2026-05-01")
        & (labels.supplier_id == "S-0417")
    ].iloc[0]
    assert may.sev3_next_90d == 1
    assert may.harm_next_90d == 302


def test_missing_severity_invalidates_targets(db):
    build_supplier_month(
        db.url.render_as_string(hide_password=False)
    )
    with db.begin() as conn:
        conn.execute(
            text("""UPDATE sqm.ncrs SET severity=NULL
            WHERE ncr_id='NCR-2026-0001'""")
        )
    labels = build_labels(db, date(2026, 9, 1))
    may = labels[
        (labels.month == "2026-05-01")
        & (labels.supplier_id == "S-0417")
    ].iloc[0]
    assert may[["sev3_next_90d", "harm_next_90d"]].isna().all()
