import pandas as pd
import pytest
from sqlalchemy import text

from sqm_ai.build1.features import build_features
from sqm_ai.features import build_supplier_month

pytestmark = pytest.mark.integration
AS_OF = pd.Timestamp("2026-09-01")


def test_deleting_future_rows_changes_nothing(db):
    build_supplier_month(
        db.url.render_as_string(hide_password=False)
    )
    full = build_features(as_of=AS_OF, engine=db)
    with db.begin() as conn:
        conn.execute(
            text("""DELETE FROM sqm.ncrs
            WHERE discovered_at >= :cutoff"""),
            {"cutoff": AS_OF},
        )
        conn.execute(
            text("""DELETE FROM sqm.audits
            WHERE audit_date >= :cutoff"""),
            {"cutoff": AS_OF},
        )
    truncated = build_features(as_of=AS_OF, engine=db)
    pd.testing.assert_frame_equal(full, truncated)
    cobalt = full[full.supplier_id.eq("S-0417")].iloc[0]
    assert cobalt.audit_score_last == 71
    assert cobalt.ncr_count_90d == 139
    assert cobalt.years_as_supplier == pytest.approx(
        11.0006844627
    )
    assert pd.isna(
        cobalt.spend_share
    )  # unavailable source, not zero


def test_exact_ninety_day_boundary_and_weighted_severity(db):
    build_supplier_month(
        db.url.render_as_string(hide_password=False)
    )
    with db.begin() as conn:
        conn.execute(
            text("""INSERT INTO sqm.ncrs VALUES
            ('B1','S-0417','material',5,0,'2026-06-03',NULL),
            ('B2','S-0417','material',5,0,'2026-06-02 23:59:59',NULL),
            ('B3','S-0417','material',5,0,'2026-09-01',NULL)""")
        )
    row = (
        build_features(AS_OF, engine=db)
        .query("supplier_id == 'S-0417'")
        .iloc[0]
    )
    assert row.ncr_count_90d == 140
    assert row.avg_severity_90d == pytest.approx(
        (92 * 2 + 118 + 5) / 140
    )
