"""LangGraph orchestration of the same bounded NCR loop."""

import argparse
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from sqm_ai.agent.replay import load_replay, restore, snapshot


class State(TypedDict):
    run: dict
    route: str


def build_graph(
    scope, call, *, checkpointer=None, interrupt_after=None
):
    def step(state):
        run = restore(state["run"], scope)
        route = run.advance(call)
        return {"run": snapshot(run), "route": route}

    graph = StateGraph(State)
    graph.add_node("step", step)
    graph.add_edge(START, "step")
    graph.add_conditional_edges(
        "step",
        lambda s: s["route"],
        {"continue": "step", "done": END},
    )
    return graph.compile(
        checkpointer=checkpointer, interrupt_after=interrupt_after
    )


def execute(run, call):
    app = build_graph(run.tools, call)
    result = app.invoke(
        {"run": snapshot(run), "route": "continue"},
        config={"recursion_limit": run.max_steps + 2},
    )
    return result["run"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/agents/replay.json"
    )
    args = parser.parse_args()
    run, call = load_replay(args.input)
    print(execute(run, call)["answer"])
