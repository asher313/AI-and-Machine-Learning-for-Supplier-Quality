"""Behavioral checks use synthetic content, injected providers, and isolated SQL."""

from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import uuid

import psycopg
import pytest
from cryptography.exceptions import InvalidTag
from sqm_ai.gateway import Gateway
from sqm_ai.gateway.audit import (
    AuditWriter,
    EncryptedContentStore,
)
from sqm_ai.gateway.policy import (
    Policy,
    GatewayUser,
    AuditUnavailable,
    BudgetExceeded,
)
from sqm_ai.gateway.router import (
    BudgetLedger,
    Endpoint,
    ModelRouter,
)


class Audit:
    def __init__(self, fail=None):
        self.events, self.fail = [], fail

    def write(self, row):
        if row["event"] == self.fail:
            raise RuntimeError("sink offline")
        self.events.append(deepcopy(row))


class Ledger:
    def __init__(self):
        self.reserved, self.settled = [], []

    def reserve(self, trace, quote):
        self.reserved.append((trace, quote))
        return "ok"

    def settle(self, trace, actual):
        self.settled.append((trace, actual))
        return actual is not None and actual <= Decimal(".02")


def build(
    tmp_path, *, response=None, failure=None, audit_fail=None
):
    calls = []

    def invoke(request):
        calls.append(request)
        if failure:
            raise failure
        return response or {
            "text": "Use the reviewed procedure [1].",
            "stop_reason": "end_turn",
            "charge": ".01",
        }

    endpoint = Endpoint(
        "synthetic-local",
        "scripted",
        frozenset({"open"}),
        invoke,
        lambda r: Decimal(".02"),
        lambda r: Decimal(r["charge"]),
    )
    audit, ledger = Audit(audit_fail), Ledger()
    store = EncryptedContentStore(
        tmp_path,
        b"x" * 32,
        key_id="test-only",
        allowed_levels={"open"},
    )
    gateway = Gateway(
        Policy(validate_citations=True),
        audit=audit,
        ledger=ledger,
        router=ModelRouter({("open", "standard"): endpoint}),
        content_store=store,
    )
    args = dict(
        prompt="Summarize",
        system="Answer from sources.",
        user=GatewayUser("fixture-user"),
        tool_name="test",
        enclave="open",
        classification="open",
        sources=["A synthetic reviewed procedure."],
    )
    return gateway, args, calls, audit, ledger, store


@pytest.mark.parametrize(
    "field,value,code",
    [
        ("system", "CUI drawing", "GW-MARK"),
        ("sources", ["Call 918-555-0142"], "GW-PII"),
        ("classification", "unknown", "GW-CLASS"),
    ],
)
def test_all_context_is_screened_before_dispatch(
    tmp_path, field, value, code
):
    g, args, calls, audit, ledger, _ = build(tmp_path)
    args[field] = value
    result = g.complete(**args)
    assert (
        result.blocked
        and code in result.codes
        and not calls
        and not ledger.reserved
    )
    assert [r["event"] for r in audit.events] == [
        "received",
        "blocked",
    ]
    assert value != audit.events[-1].get("prompt")


def test_archive_reconstructs_actual_request_and_raw_response(
    tmp_path,
):
    g, args, calls, audit, ledger, store = build(
        tmp_path,
        response={
            "text": "Write to a@example.test [1].",
            "stop_reason": "end_turn",
            "charge": ".01",
        },
    )
    result = g.complete(**args)
    assert (
        not result.blocked and "[EMAIL REDACTED]" in result.text
    )
    row = audit.events[-1]
    raw = store.get(
        row["response_ref"],
        trace_id=result.trace_id,
        kind="response",
        classification="open",
    )
    request = store.get(
        row["request_ref"],
        trace_id=result.trace_id,
        kind="request",
        classification="open",
    )
    assert (
        json.loads(request) == calls[0]
        and args["sources"][0] in request
    )
    assert (
        "a@example.test" in raw
        and row["display_hash"] != row["raw_response_hash"]
    )
    assert [r["event"] for r in audit.events] == [
        "received",
        "dispatch_intent",
        "completed",
    ]
    assert ledger.settled[0][1] == Decimal(".01")


def test_higher_marked_response_never_enters_lower_archive(
    tmp_path,
):
    g, args, calls, audit, ledger, store = build(
        tmp_path,
        response={
            "text": "CUI drawing [1]",
            "stop_reason": "end_turn",
            "charge": ".01",
        },
    )
    result = g.complete(**args)
    assert result.blocked and "GW-POST-MARK" in result.codes
    assert (
        audit.events[-1]["archive_gap"]
        == "response_requires_higher_approved_storage"
    )
    assert (
        len(list(tmp_path.glob("*.json"))) == 1
    )  # Request only.
    assert ledger.settled[0][1] == Decimal(".01")


@pytest.mark.parametrize(
    "fail", ["received", "dispatch_intent", "completed"]
)
def test_audit_failure_suppresses_output(tmp_path, fail):
    g, args, calls, audit, ledger, _ = build(
        tmp_path, audit_fail=fail
    )
    if fail == "received":
        with pytest.raises(AuditUnavailable):
            g.complete(**args)
    else:
        result = g.complete(**args)
        assert result.blocked and result.codes == ["GW-AUDIT"]
    assert len(calls) == (1 if fail == "completed" else 0)
    if fail == "dispatch_intent":
        assert ledger.settled[0][1] == 0


def test_timeout_keeps_reservation_and_never_leaks_exception(
    tmp_path,
):
    g, args, calls, audit, ledger, _ = build(
        tmp_path, failure=TimeoutError("secret payload")
    )
    result = g.complete(**args)
    assert (
        result.blocked
        and len(calls) == 1
        and ledger.settled[0][1] is None
    )
    assert (
        "secret payload" not in json.dumps(audit.events)
        and "secret payload" not in result.text
    )


@pytest.mark.parametrize(
    "response,code",
    [
        (
            {
                "text": "Unsupported [2]",
                "stop_reason": "end_turn",
                "charge": ".01",
            },
            "GW-CITE",
        ),
        (
            {
                "text": "partial [1]",
                "stop_reason": "max_tokens",
                "charge": ".01",
            },
            "GW-INCOMPLETE",
        ),
        (
            {
                "text": "Answer [1]",
                "stop_reason": "end_turn",
                "charge": ".03",
            },
            "GW-QUOTE",
        ),
    ],
)
def test_failed_outputs_are_withheld_but_cost_is_settled(
    tmp_path, response, code
):
    g, args, calls, audit, ledger, _ = build(
        tmp_path, response=response
    )
    result = g.complete(**args)
    assert (
        result.blocked
        and result.codes == [code]
        and ledger.settled[0][1] == Decimal(response["charge"])
    )


def test_encryption_context_wrong_key_and_tampering(tmp_path):
    store = EncryptedContentStore(
        tmp_path,
        b"a" * 32,
        key_id="test",
        allowed_levels={"open"},
    )
    ref = store.put(
        "trace",
        "request",
        "private fixture",
        classification="open",
    )
    args = dict(
        trace_id="trace", kind="request", classification="open"
    )
    assert store.get(ref, **args) == "private fixture"
    with pytest.raises(ValueError):
        store.get(ref, **dict(args, trace_id="different"))
    wrong = EncryptedContentStore(
        tmp_path,
        b"b" * 32,
        key_id="test",
        allowed_levels={"open"},
    )
    with pytest.raises(InvalidTag):
        wrong.get(ref, **args)
    path = tmp_path / (ref + ".json")
    blob = json.loads(path.read_text())
    blob["ciphertext"] = (
        f"{int(blob['ciphertext'][:2], 16) ^ 1:02x}"
        + blob["ciphertext"][2:]
    )
    path.write_text(json.dumps(blob))
    with pytest.raises(InvalidTag):
        store.get(ref, **args)


@pytest.fixture
def gateway_dsn(db):
    dsn = db.url.set(drivername="postgresql").render_as_string(
        hide_password=False
    )
    with psycopg.connect(dsn) as conn:
        conn.execute(
            (
                Path(__file__).parents[2]
                / "src/sqm_ai/gateway/schema.sql"
            ).read_text()
        )
    return dsn


def test_concurrent_reservations_reconciliation_and_consistent_cap(
    gateway_dsn,
):
    ledger = BudgetLedger(gateway_dsn, 1, 0.8)

    def reserve(_):
        trace = str(uuid.uuid4())
        try:
            ledger.reserve(trace, Decimal(".4"))
            return trace
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        traces = [x for x in pool.map(reserve, range(8)) if x]
    assert len(traces) == 2
    ledger.settle(traces[0], None)
    with pytest.raises(ValueError):
        ledger.settle(traces[0], Decimal(".1"))
    ledger.settle(traces[0], Decimal(".1"), reconcile=True)
    ledger.settle(
        traces[0], Decimal(".1")
    )  # Idempotent acknowledgement retry.
    with pytest.raises(ValueError):
        ledger.settle(traces[0], Decimal(".2"))
    with pytest.raises(ValueError):
        BudgetLedger(gateway_dsn, 2, 0.8).reserve(
            str(uuid.uuid4()), ".1"
        )
    ledger.settle(traces[1], Decimal(".000000001"))
    with psycopg.connect(gateway_dsn) as conn:
        spent, reserved = conn.execute(
            "SELECT spent,reserved FROM sqm.gateway_budget_months"
        ).fetchone()
    assert spent == Decimal(".10000001") and reserved == 0


def test_audit_commits_outside_business_rollback(gateway_dsn):
    trace = str(uuid.uuid4())
    with psycopg.connect(gateway_dsn) as conn:
        AuditWriter(gateway_dsn).write(
            {
                "trace_id": trace,
                "event": "received",
                "input_hash": "synthetic",
            }
        )
        conn.rollback()
        row = conn.execute(
            "SELECT record FROM sqm.llm_audit_events WHERE trace_id=%s",
            (trace,),
        ).fetchone()[0]
    assert row["input_hash"] == "synthetic"


def test_endpoint_label_does_not_grant_user_clearance(tmp_path):
    g, args, calls, audit, ledger, store = build(tmp_path)
    args.update(classification="controlled", enclave="controlled")
    result = g.complete(**args)
    assert result.codes == ["GW-AUTH"] and not calls
    args["user"] = GatewayUser(
        "cleared", allowed_levels=frozenset({"controlled"})
    )
    result = g.complete(**args)
    assert result.codes == ["GW-ROUTE"] and not calls


def test_higher_capacity_endpoint_does_not_raise_output_authorization(
    tmp_path,
):
    g, args, calls, audit, ledger, store = build(
        tmp_path,
        response={
            "text": "CUI result [1]",
            "stop_reason": "end_turn",
            "charge": ".01",
        },
    )
    old = g.router.endpoints[("open", "standard")]
    g.router = ModelRouter(
        {
            ("controlled", "standard"): Endpoint(
                old.name,
                old.model,
                frozenset({"controlled"}),
                old.invoke,
                old.quote,
                old.charge,
            )
        }
    )
    store.allowed_levels = frozenset({"open", "controlled"})
    args["enclave"] = "controlled"
    result = g.complete(**args)
    assert result.codes == ["GW-POST-MARK"] and result.blocked
