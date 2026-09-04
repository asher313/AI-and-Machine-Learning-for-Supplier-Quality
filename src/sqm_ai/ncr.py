# src/sqm_ai/ncr.py
import re
from datetime import datetime

from pydantic import BaseModel, Field, field_validator

NCR_ID = re.compile(r"^NCR-\d{4}-\d{4}$")
SUPPLIER_ID = re.compile(r"^S-\d{4}$")


class NCRInput(BaseModel):
    """An NCR from SAP, validated at the boundary."""

    ncr_id: str
    supplier_id: str
    defect_description: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    discovered_at: datetime

    @field_validator("ncr_id")
    @classmethod
    def ncr_id_format(cls, v: str) -> str:
        if not NCR_ID.match(v):
            raise ValueError("must look like NCR-2026-0042")
        return v

    @field_validator("supplier_id")
    @classmethod
    def supplier_id_format(cls, v: str) -> str:
        if not SUPPLIER_ID.match(v):
            raise ValueError("supplier_id must look like S-0417")
        return v
