"""Exercise the real PostgreSQL rollup on a disposable fixture."""

import pandas as pd
import pytest

from sqm_ai.features import build_supplier_month

pytestmark = pytest.mark.integration


@pytest.fixture
def sm(db):
    return build_supplier_month(
        db.url.render_as_string(hide_password=False)
    )


def test_one_row_per_eligible_supplier_month(sm):
    # Twelve months for Cobalt, three after the second supplier joins.
    assert len(sm) == 15
    assert not sm.duplicated(["supplier_id", "month"]).any()
    assert sm.loc[
        sm.supplier_id.eq("S-0002"), "month"
    ].min() == pd.Timestamp("2026-06-01")


def test_rates_and_empty_activity(sm):
    for col in ["fpy", "otd"]:
        assert sm[col].dropna().between(0, 1).all()
    absent = sm[sm.supplier_id.eq("S-0002")]
    assert absent.ncr_count.eq(0).all()
    assert absent.fpy.isna().all()


def test_cobalt_august_respects_cutoff(sm):
    row = sm[
        sm.supplier_id.eq("S-0417") & sm.month.eq("2026-08-01")
    ].iloc[0]
    assert (
        row.ncr_count,
        row.sev3_plus_count,
        row.ncr_count_3mo,
    ) == (47, 9, 139)
    assert row.fpy == pytest.approx(0.912)
    assert row.otd == pytest.approx(0.83)
    assert row.audit_score == 71  # September audit must not leak.
    assert row.open_cars == 4


def test_refresh_preserves_view_and_supports_full_history(db):
    from sqlalchemy import text

    url = db.url.render_as_string(hide_password=False)
    build_supplier_month(url)
    with db.begin() as conn:
        conn.execute(
            text(
                "CREATE VIEW sqm.month_view AS SELECT * FROM sqm.supplier_month"
            )
        )
    full = build_supplier_month(url, "2023-09-01", "2026-08-01")
    assert len(full) == 39
    with db.connect() as conn:
        assert (
            conn.execute(
                text("SELECT COUNT(*) FROM sqm.month_view")
            ).scalar_one()
            == 39
        )
