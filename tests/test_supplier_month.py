# tests/test_supplier_month.py
import pandas as pd
import pytest

from sqm_ai.settings import get_settings
from sqm_ai.features import build_supplier_month

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sm() -> pd.DataFrame:
    return build_supplier_month(get_settings().database_url)


def test_one_row_per_supplier_month(sm):
    assert len(sm) == 21_600
    assert not sm.duplicated(["supplier_id", "month"]).any()


def test_rates_are_fractions(sm):
    for col in ["fpy", "otd"]:
        s = sm[col].dropna()
        assert ((s >= 0) & (s <= 1)).all(), col


def test_no_future_months(sm):
    assert sm["month"].max() == pd.Timestamp("2026-08-01")


def test_cobalt_august(sm):
    row = sm.query(
        "supplier_id == 'S-0417' and month == '2026-08-01'"
    ).iloc[0]
    assert row["ncr_count"] == 47
    assert row["sev3_plus_count"] == 9
