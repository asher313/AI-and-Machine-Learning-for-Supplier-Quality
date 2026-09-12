import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest
from anthropic.types import Message

from sqm_ai.agent.loop import AgentRun, AgentStalled, run_agent
from sqm_ai.agent.replay import load_replay
from sqm_ai.agent.tools import AgentTools

ROOT = Path(__file__).resolve().parents[1]


def module(path):
    import sys

    name = path.stem + "_test"
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def replay(tmp_path):
    generator = module(ROOT / "scripts/generate_agent_data.py")
    p = tmp_path / "replay.json"
    p.write_text(json.dumps(generator.generate()))
    return p


def test_scope_policy_and_proposal_invalidation(replay):
    run, call = load_replay(replay)
    tools = run.tools
    data = json.loads(replay.read_text())
    triage = data["responses"][0]["content"][0]["input"]
    car = data["responses"][2]["content"][0]["input"]
    with pytest.raises(ValueError):
        tools.execute(
            "get_supplier_history", {"supplier_id": "S-0001"}
        )
    with pytest.raises(ValueError):
        tools.execute("draft_car", car)
    tools.execute("record_triage", triage | {"severity": 2})
    tools.execute(
        "get_supplier_history", {"supplier_id": "S-0417"}
    )
    with pytest.raises(ValueError):
        tools.execute("draft_car", car)
    tools.execute("record_triage", triage)
    first = tools.execute("draft_car", car)
    assert (
        tools.execute("draft_car", car)["proposal_id"]
        == first["proposal_id"]
    )
    tools.execute("record_triage", triage | {"severity": 4})
    assert tools.car is None
    with pytest.raises(ValueError):
        tools.require_complete()


def test_loop_completion_caps_and_error_redaction(replay):
    run, call = load_replay(replay)
    assert "Propose" in run_agent(run, call=call)
    assert run.step == 4 and run.tool_calls == 3
    assert all(not e["failed"] for e in run.tools.events)
    run, call = load_replay(replay)
    run.max_steps = 1
    with pytest.raises(AgentStalled):
        run_agent(run, call=call)
    run, call = load_replay(replay)
    response = call(run.messages)
    response.stop_reason = "max_tokens"
    with pytest.raises(AgentStalled):
        run.advance(lambda *a, **k: response)
    assert run.tools.triage is None
    run, call = load_replay(replay)
    response = call(run.messages)
    response.content[0].name = "unknown"
    run.advance(lambda *a, **k: response)
    result = run.messages[-1]["content"][0]
    assert (
        result["is_error"]
        and "Tool rejected" in result["content"]
    )
    assert run.tools.triage is None


def test_multi_tool_results_are_adjacent_and_batch_budget_atomic(
    replay,
):
    run, call = load_replay(replay)
    data = json.loads(replay.read_text())
    response = Message.model_validate(data["responses"][0])
    response.content.append(
        Message.model_validate(data["responses"][1]).content[0]
    )
    run.max_tool_calls = 1
    with pytest.raises(AgentStalled):
        run.advance(lambda *a, **k: response)
    assert run.tools.triage is None
    run.max_tool_calls = 2
    run.advance(lambda *a, **k: response)
    assert len(run.messages[-1]["content"]) == 2
    assert run.messages[-2]["role"] == "assistant"


def test_frameworks_replay_same_tools_and_langgraph_restores(
    replay, tmp_path
):
    import asyncio
    import os

    os.environ["CREWAI_TELEMETRY_DISABLED"] = "true"
    os.environ["OTEL_SDK_DISABLED"] = "true"
    os.environ["CREWAI_STORAGE_DIR"] = str(tmp_path / "crewai")
    pytest.importorskip("langgraph")
    pytest.importorskip("autogen_core")
    pytest.importorskip("crewai")
    from langgraph.checkpoint.sqlite import SqliteSaver
    from sqm_ai.agent.replay import snapshot

    lg = module(ROOT / "examples/agents/langgraph_ncr.py")
    ag = module(ROOT / "examples/agents/autogen_ncr.py")
    crew = module(ROOT / "examples/agents/crewai_ncr.py")
    results = []
    for fn in [
        lg.execute,
        lambda r, c: asyncio.run(ag.execute(r, c)),
        crew.execute,
    ]:
        run, call = load_replay(replay)
        results.append(fn(run, call))
    assert all(
        x["step"] == 4 and x["tool_calls"] == 3 for x in results
    )
    assert len({x["answer"] for x in results}) == 1
    assert len({x["car"]["proposal_id"] for x in results}) == 1
    database = str(tmp_path / "checkpoint.sqlite")
    config = {
        "configurable": {"thread_id": "synthetic-review"},
        "recursion_limit": 12,
    }
    run, call = load_replay(replay)
    with SqliteSaver.from_conn_string(database) as saver:
        app = lg.build_graph(
            run.tools,
            call,
            checkpointer=saver,
            interrupt_after=["step"],
        )
        first = app.invoke(
            {"run": snapshot(run), "route": "continue"}, config
        )
        assert first["run"]["step"] == 1
    # New connection and graph prove persisted state, not the old Python object.
    run, call = load_replay(replay)
    with SqliteSaver.from_conn_string(database) as saver:
        app = lg.build_graph(run.tools, call, checkpointer=saver)
        restored = app.invoke(None, config)
    assert restored["run"]["step"] == 4
    assert (
        restored["run"]["car"]["proposal_id"]
        == results[0]["car"]["proposal_id"]
    )


def test_native_sdk_loop_uses_complete_tool_protocol(
    replay, monkeypatch
):
    import anthropic
    import httpx2
    import sqm_ai.llm as llm
    import sqm_ai.agent.loop as loop

    data = json.loads(replay.read_text())
    requests = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path.endswith("count_tokens"):
            return httpx2.Response(
                200, json={"input_tokens": 1000}
            )
        step = len(requests)
        requests.append(body)
        assert len(body["tools"]) == 3
        assert "temperature" not in body
        if step:
            previous = body["messages"][-2]["content"]
            results = body["messages"][-1]["content"]
            assert [
                b["id"]
                for b in previous
                if b["type"] == "tool_use"
            ] == [b["tool_use_id"] for b in results]
        response = data["responses"][step] | {
            "model": llm.MODELS["standard"]
        }
        return httpx2.Response(200, json=response)

    with httpx2.Client(
        transport=httpx2.MockTransport(handler)
    ) as http:
        client = anthropic.Anthropic(
            api_key="test-only", http_client=http, max_retries=0
        )
        monkeypatch.setattr(llm, "client", client)
        monkeypatch.setattr(loop, "client", client)
        run, _ = load_replay(replay)
        assert "Propose" in loop.run_agent(run)
    assert len(requests) == 4


def test_history_count_and_asof_use_all_rows_but_only_ten_examples(
    db,
):
    from sqlalchemy import text
    from sqm_ai.agent.tools import database_history

    with db.begin() as conn:
        conn.execute(text("DELETE FROM sqm.ncrs"))
        records = [
            {
                "id": f"prior-{i}",
                "sid": "S-0417",
                "date": "2026-08-01",
            }
            for i in range(12)
        ]
        records += [
            {
                "id": "current",
                "sid": "S-0417",
                "date": "2026-08-18",
            },
            {
                "id": "future",
                "sid": "S-0417",
                "date": "2026-09-01",
            },
            {"id": "old", "sid": "S-0417", "date": "2026-01-01"},
            {
                "id": "other",
                "sid": "S-0002",
                "date": "2026-08-01",
            },
        ]
        conn.execute(
            text(
                "INSERT INTO sqm.ncrs (ncr_id,supplier_id,category,severity,discovered_at) VALUES (:id,:sid,'dimensional',2,CAST(:date AS timestamp))"
            ),
            records,
        )
    result = database_history(db)(
        "S-0417", dt.datetime(2026, 8, 18, 10), "current"
    )
    assert result["total_in_window"] == 12
    assert len(result["recent"]) == 10
