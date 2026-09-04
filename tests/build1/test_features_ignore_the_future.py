# tests/build1/test_features_ignore_the_future.py
import pandas as pd

from sqm_ai.build1.features import build_features

AS_OF = pd.Timestamp("2026-05-01")


def test_deleting_future_rows_changes_nothing(db):
    """Features must not move when the future is gone."""
    full = build_features(as_of=AS_OF, engine=db)
    db.execute(
        "DELETE FROM sqm.ncrs WHERE discovered_at >= %s",
        (AS_OF,))
    db.execute(
        "DELETE FROM sqm.audits WHERE audit_date >= %s",
        (AS_OF,))
    truncated = build_features(as_of=AS_OF, engine=db)
    pd.testing.assert_frame_equal(
        full.sort_values("supplier_id"),
        truncated.sort_values("supplier_id"),
        check_exact=False, rtol=1e-9,
    )
