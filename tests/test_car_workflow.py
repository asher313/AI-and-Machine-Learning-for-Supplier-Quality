from copy import deepcopy
import json
from pathlib import Path

import pytest

from sqm_ai.car.redact import Redactor
from sqm_ai.car.root_cause import RootCauseSet, Why

ROOT = Path(__file__).resolve().parents[1]


def load_module(path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def services(tmp_path):
    pytest.importorskip("langgraph.checkpoint.sqlite")
    from sqm_ai.car.run_build5 import replay_services

    generator = load_module(ROOT / "scripts/generate_car_data.py")
    path = tmp_path / "car.json"
    path.write_text(json.dumps(generator.generate()))
    return replay_services(path)


def test_redaction_is_case_aware_collision_safe_and_restores_names():
    names = {
        f"S-{i:04d}": f"Supplier Name {i}" for i in range(30)
    }
    names["S-0417"] = "Cobalt Machining"
    r = Redactor.build(names, {"S-0417": ["Cobalt Mach."]})
    raw = "cobalt machining and COBALT MACH. [S-0417]; Supplier Name 1, Supplier Name 10"
    out = r.out(raw)
    assert not r.leaked(out)
    assert r.back(out).count("Cobalt Machining") == 3
    assert "Supplier Name 10" in r.back(out)
    with pytest.raises(ValueError):
        r.out("forged [[SUPPLIER_0001]]")
    with pytest.raises(ValueError):
        r.back("[[SUPPLIER_9999]]")
    with pytest.raises(ValueError):
        Redactor.build({"S-0001": "SAME", "S-0002": "same"})


def test_unknown_links_do_not_need_fabricated_sources():
    why = Why(
        question="Why did this occur?",
        answer="UNKNOWN",
        status="unknown",
        evidence_refs=[],
        investigation_gap="Inspect the fixture and compare location under controlled conditions.",
    )
    assert not why.evidence_refs
    with pytest.raises(ValueError):
        Why(
            question="Why did this occur?",
            answer="UNKNOWN",
            status="unknown",
            evidence_refs=["invented"],
            investigation_gap="Inspect it.",
        )
    with pytest.raises(ValueError):
        RootCauseSet(hypotheses=[], investigation_gaps=[])


def test_review_persists_and_requires_authorized_explicit_resume(
    services, tmp_path
):
    from langgraph.types import Command
    from sqm_ai.car.graph import build_graph

    config = {
        "configurable": {"thread_id": "test-car"},
        "recursion_limit": 64,
    }
    path = tmp_path / "review.sqlite"
    with build_graph(path, services) as app:
        result = app.invoke(
            {
                "ncr_id": services.ncr_id,
                "supplier_id": services.supplier_id,
            },
            config,
        )
        assert app.get_state(config).next == ("review",)
        assert not result["failures"] and not result["gaps"]
        assert result["draft"].count("## ") == 7
        assert "Cobalt Machining" in result["draft"]
        assert "Cobalt Machining" not in json.dumps(
            result["evidence"]
        )
    with build_graph(path, services) as app:
        decision = {
            "actor_id": "synthetic-reviewer",
            "action": "accept_draft",
            "note": "Reviewed this synthetic fixture",
            "evidence": None,
        }
        final = app.invoke(Command(resume=decision), config)
        assert final["status"] == "accepted_draft"
        assert not app.get_state(config).next


def test_invalid_reference_repairs_are_bounded_per_role(
    services, tmp_path
):
    from sqm_ai.car.graph import build_graph

    original, calls = services.role_caller, []

    def bad(role, context, **kwargs):
        result = original(role, context, **kwargs)
        calls.append(role)
        if role == "problem":
            result["evidence_refs"] = ["missing-1", "missing-2"]
        return result

    services.role_caller = bad
    with build_graph(tmp_path / "bad.sqlite", services) as app:
        result = app.invoke(
            {"ncr_id": services.ncr_id},
            {
                "configurable": {"thread_id": "bad"},
                "recursion_limit": 64,
            },
        )
    assert calls.count("problem") == 3
    assert result["retries"]["problem"] == 2
    assert result["failures"]


def test_gap_blocks_acceptance_and_new_evidence_reenters_research(
    services, tmp_path
):
    from langgraph.types import Command
    from sqm_ai.car.graph import build_graph

    original = services.researcher

    def missing(state):
        result = original(state)
        result["investigation_gaps"] = [
            "Obtain the inspection record."
        ]
        return result

    services.researcher = missing
    config = {
        "configurable": {"thread_id": "gap"},
        "recursion_limit": 64,
    }
    with build_graph(tmp_path / "gap.sqlite", services) as app:
        first = app.invoke({"ncr_id": services.ncr_id}, config)
        assert first["gaps"]
        decision = {
            "actor_id": "synthetic-reviewer",
            "action": "provide_evidence",
            "note": "Reviewed corrected source bundle",
            "evidence": original({}),
        }
        second = app.invoke(Command(resume=decision), config)
        assert not second["gaps"] and app.get_state(
            config
        ).next == ("review",)
        assert len(second["reviews"]) == 1


def test_unauthorized_review_and_unresolved_gaps_cannot_be_accepted(
    services, tmp_path
):
    from langgraph.types import Command
    from sqm_ai.car.graph import build_graph

    original = services.researcher

    def missing(state):
        result = original(state)
        result["investigation_gaps"] = [
            "Required measurement is absent."
        ]
        return result

    services.researcher = missing
    decision = {
        "actor_id": "intruder",
        "action": "accept_draft",
        "note": "Trying to bypass review",
        "evidence": None,
    }
    config = {
        "configurable": {"thread_id": "unauthorized"},
        "recursion_limit": 64,
    }
    with build_graph(
        tmp_path / "unauthorized.sqlite", services
    ) as app:
        app.invoke({"ncr_id": services.ncr_id}, config)
        with pytest.raises(PermissionError):
            app.invoke(Command(resume=decision), config)
    config = {
        "configurable": {"thread_id": "gapped"},
        "recursion_limit": 64,
    }
    with build_graph(tmp_path / "gapped.sqlite", services) as app:
        app.invoke({"ncr_id": services.ncr_id}, config)
        decision["actor_id"] = "synthetic-reviewer"
        with pytest.raises(ValueError, match="resolve failures"):
            app.invoke(Command(resume=decision), config)


def test_recurrence_has_one_denominator_row_per_mature_car(db):
    from sqlalchemy import text

    with db.begin() as conn:
        sql = (ROOT / "sql/car_outcomes.sql").read_text()
        for statement in sql.split(";"):
            if statement.strip():
                conn.execute(text(statement))
        conn.execute(
            text(
                "INSERT INTO sqm.car_outcomes VALUES ('C1','S-0417','fixture','2026-01-01'), ('C2','S-0002','material','2026-01-01'), ('C3','S-0417','fixture','2026-08-01'), ('C4','S-0417',NULL,'2026-01-01')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sqm.car_final_reviews SELECT car_id,'reviewer' FROM sqm.car_outcomes"
            )
        )
        conn.execute(
            text(
                "INSERT INTO sqm.ncr_recurrence_events VALUES ('R1','S-0417','fixture','2026-02-01','2026-02-02'), ('R2','S-0417','fixture','2026-03-01','2026-03-02'), ('R3','S-0002','material','2026-03-01','2026-10-01')"
            )
        )
        result = (
            conn.execute(
                text(
                    (ROOT / "sql/car_recurrence.sql").read_text()
                ),
                {"asof": "2026-09-01"},
            )
            .mappings()
            .one()
        )
        assert result["eligible_cars"] == 2
        assert result["cars_with_recurrence"] == 1
        assert float(result["recurrence_rate"]) == 0.5


def test_memory_is_persistent_scoped_and_keeps_summary_provenance(
    tmp_path,
):
    from sqm_ai.assistant.access import AssistantUser
    from sqm_ai.car.memory import CarMemory

    scope = AssistantUser("u1", ("NL-KESTREL",))
    seen = []

    def retrieve(query, user, *, asof, keep):
        seen.append((user, asof, keep))
        return []

    def summarize(notes):
        assert all(n["review_state"] == "proposal" for n in notes)
        return "A prior proposal exists; consult the linked source. No root cause is confirmed."

    path = str(tmp_path / "memory.sqlite")
    memory = CarMemory(
        "S-0417",
        scope,
        path,
        summarize,
        "policy-v1",
        max_working=2,
        retriever=retrieve,
    )
    first = memory.add(
        "Candidate fixture cause is unconfirmed",
        source_ids=["N1"],
    )
    memory.add("Investigation is planned", source_ids=["N2"])
    memory.add("Review is pending", source_ids=["N3"])
    restored = CarMemory(
        "S-0417",
        scope,
        path,
        summarize,
        "policy-v1",
        max_working=2,
        retriever=retrieve,
    )
    result = restored.recall("fixture")
    assert (
        len(result["working"]) == 2
        and len(result["episodic"]) == 1
    )
    assert result["episodic"][0]["source_event_ids"] == [first]
    assert (
        result["episodic"][0]["status"]
        == "unverified_derived_summary"
    )
    other = CarMemory(
        "S-0002",
        scope,
        path,
        summarize,
        "policy-v1",
        retriever=retrieve,
    )
    assert not other.recall("fixture")["working"]
    revoked = CarMemory(
        "S-0417",
        AssistantUser("u1", ()),
        path,
        summarize,
        "policy-v1",
        retriever=retrieve,
    )
    assert not revoked.recall("fixture")["episodic"]
