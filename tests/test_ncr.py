# tests/test_ncr.py
import pytest
from pydantic import ValidationError

from sqm_ai.ncr import NCRInput


@pytest.fixture
def reference_ncr() -> dict:
    return {
        "ncr_id": "NCR-2026-0042",
        "supplier_id": "S-0417",
        "defect_description": "hole position 2 mm out of tol",
        "quantity": 12,
        "discovered_at": "2026-08-18T09:14:00",
    }


def test_reference_ncr_validates(reference_ncr):
    ncr = NCRInput(**reference_ncr)
    assert ncr.quantity == 12
    assert ncr.discovered_at.year == 2026


@pytest.mark.parametrize(
    "bad_id", ["2026-0042", "ncr-2026-0042", "NCR-42", ""]
)
def test_bad_ncr_id_rejected(reference_ncr, bad_id):
    reference_ncr["ncr_id"] = bad_id
    with pytest.raises(ValidationError):
        NCRInput(**reference_ncr)


def test_zero_quantity_rejected(reference_ncr):
    reference_ncr["quantity"] = 0
    with pytest.raises(ValidationError):
        NCRInput(**reference_ncr)
