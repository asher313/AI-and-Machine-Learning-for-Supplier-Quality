# tests/test_extract_contract.py
import pytest

from sqm_ai.extract import SchemaContractError, _check

WANT = ["supplier_id", "supplier_name", "tier", "active"]


def test_contract_passes_when_columns_present():
    _check("V_SUPPLIERS", WANT + ["changed_at"], WANT)


def test_contract_names_the_missing_column():
    with pytest.raises(SchemaContractError) as e:
        _check("V_SUPPLIERS", ["supplier_id", "status"], WANT)
    assert "active" in str(e.value)
