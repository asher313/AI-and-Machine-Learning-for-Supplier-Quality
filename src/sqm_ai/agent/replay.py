"""Synthetic model responses for protocol tests; never a quality benchmark."""

import datetime as dt
from copy import deepcopy
import json
from pathlib import Path

from anthropic.types import Message

from sqm_ai.agent.loop import AgentRun
from sqm_ai.agent.tools import AgentTools


def load_replay(path):
    data = json.loads(Path(path).read_text())
    if data.get("synthetic") is not True:
        raise ValueError(
            "explicit synthetic replay fixture required"
        )
    ncr = data["ncr"]

    def history(supplier_id, asof, ncr_id):
        if (
            supplier_id != ncr["supplier_id"]
            or ncr_id != ncr["ncr_id"]
        ):
            raise ValueError("fixture scope mismatch")
        return data["history"]

    tools = AgentTools(
        ncr["ncr_id"],
        ncr["supplier_id"],
        dt.datetime.fromisoformat(ncr["asof"]),
        history,
    )
    run = AgentRun(
        tools, ncr["description"], data_classification="synthetic"
    )

    def call(messages, **kwargs):
        # Derive position from transcript, so a restored run needs no hidden cursor.
        step = sum(m["role"] == "assistant" for m in messages)
        if step >= len(data["responses"]):
            raise ValueError(
                "synthetic response script exhausted"
            )
        return Message.model_validate(data["responses"][step])

    return run, call


def snapshot(run):
    return deepcopy(
        {
            "description": run.description,
            "data_classification": run.data_classification,
            "max_steps": run.max_steps,
            "max_tool_calls": run.max_tool_calls,
            "step": run.step,
            "tool_calls": run.tool_calls,
            "answer": run.answer,
            "messages": run.messages,
            "triage": run.tools.triage.model_dump()
            if run.tools.triage
            else None,
            "history": run.tools.history,
            "car": run.tools.car,
            "events": run.tools.events,
        }
    )


def restore(state, scope):
    """Resume only within a server-supplied NCR scope, never browser-provided permissions."""
    from sqm_ai.triage.schema import TriageResult

    state = deepcopy(state)

    tools = AgentTools(
        scope.ncr_id,
        scope.supplier_id,
        scope.asof,
        scope.history_reader,
        triage=TriageResult.model_validate(state["triage"])
        if state["triage"]
        else None,
        history=state["history"],
        car=state["car"],
        events=state["events"],
    )
    keys = (
        "description",
        "data_classification",
        "max_steps",
        "max_tool_calls",
        "step",
        "tool_calls",
        "answer",
        "messages",
    )
    return AgentRun(tools, **{k: state[k] for k in keys})
