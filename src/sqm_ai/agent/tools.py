"""Run-scoped proposals and read-only history; no operational writes."""

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field

from anthropic import transform_schema
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import text

from sqm_ai.triage.classify import TRIAGE_TOOL
from sqm_ai.triage.schema import TriageResult

HISTORY_SQL = """
SELECT category, severity, discovered_at,
       count(*) OVER () AS total_in_window
  FROM sqm.ncrs
 WHERE supplier_id = :sid
   AND discovered_at >= :start
   AND discovered_at < :asof
   AND ncr_id <> :ncr_id
 ORDER BY discovered_at DESC, ncr_id
 LIMIT 10
"""


class HistoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    supplier_id: str = Field(pattern=r"^S-\d{4}$")


class CarArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ncr_id: str = Field(min_length=1)
    supplier_id: str = Field(pattern=r"^S-\d{4}$")
    root_cause_hypothesis: str = Field(
        min_length=1, max_length=1000
    )
    proposed_containment: str = Field(
        min_length=1, max_length=1000
    )


HISTORY_TOOL = {
    "name": "get_supplier_history",
    "description": "Read authorized prior 90-day history for the current supplier: total count and at most ten recent examples.",
    "strict": True,
    "input_schema": transform_schema(HistoryArgs),
}
CAR_TOOL = {
    "name": "draft_car",
    "description": "Prepare an in-memory CAR proposal after triage and history, only for severity >=3. Does not open or send a CAR.",
    "strict": True,
    "input_schema": transform_schema(CarArgs),
}
TOOLS = [TRIAGE_TOOL, HISTORY_TOOL, CAR_TOOL]


def database_history(engine):
    """Inject a read-only connection whose source access was approved by the server."""

    def read(supplier_id, asof, ncr_id):
        with engine.connect() as conn:
            rows = (
                conn.execute(
                    text(HISTORY_SQL),
                    {
                        "sid": supplier_id,
                        "ncr_id": ncr_id,
                        "start": asof - dt.timedelta(days=90),
                        "asof": asof,
                    },
                )
                .mappings()
                .all()
            )
        return {
            "total_in_window": int(rows[0]["total_in_window"])
            if rows
            else 0,
            "recent": [
                {
                    k: v
                    for k, v in row.items()
                    if k != "total_in_window"
                }
                for row in rows
            ],
            "asof": asof.isoformat(),
        }

    return read


@dataclass
class AgentTools:
    """One trusted NCR/supplier scope per run; never share across users."""

    ncr_id: str
    supplier_id: str
    asof: dt.datetime
    history_reader: object
    triage: TriageResult | None = None
    history: dict | None = None
    car: dict | None = None
    events: list = field(default_factory=list)

    def execute(self, name, arguments):
        if name == "record_triage":
            result = TriageResult.model_validate(arguments)
            self._supplier(result.supplier_id)
            self.triage = result
            self.car = None  # A revised triage invalidates an earlier draft.
            return {
                "status": "proposal_in_memory",
                "triage": result.model_dump(),
            }
        if name == "get_supplier_history":
            args = HistoryArgs.model_validate(arguments)
            self._supplier(args.supplier_id)
            result = self.history_reader(
                self.supplier_id, self.asof, self.ncr_id
            )
            if (
                not isinstance(result, dict)
                or not isinstance(
                    result.get("total_in_window"), int
                )
                or result["total_in_window"] < 0
            ):
                raise ValueError("invalid history response")
            self.history = result
            return result
        if name == "draft_car":
            args = CarArgs.model_validate(arguments)
            self._supplier(args.supplier_id)
            if args.ncr_id != self.ncr_id:
                raise ValueError("NCR outside this run")
            if self.triage is None or self.history is None:
                raise ValueError(
                    "triage and history required before CAR proposal"
                )
            if self.triage.severity < 3:
                raise ValueError(
                    "CAR proposal requires severity >=3"
                )
            payload = args.model_dump() | {
                "triage": self.triage.model_dump()
            }
            proposal_id = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            self.car = {
                "proposal_id": proposal_id,
                "status": "human_review_required",
                **payload,
            }
            return self.car
        raise ValueError("tool is not allowlisted")

    def _supplier(self, supplier_id):
        if supplier_id != self.supplier_id:
            raise ValueError("supplier outside this run")

    def require_complete(self):
        if self.triage is None or self.history is None:
            raise ValueError("triage and history are incomplete")
        if self.triage.severity >= 3 and self.car is None:
            raise ValueError("required CAR proposal is missing")
