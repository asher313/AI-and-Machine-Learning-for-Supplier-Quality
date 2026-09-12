"""Bounded repairs plus a durable human decision on every finished draft."""

from contextlib import contextmanager

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from sqm_ai.car.nodes import make_nodes
from sqm_ai.car.state import CarState
from sqm_ai.car.validate import validate_node

MAX_RETRIES = 2  # Two additional repair attempts after the original role call.


def route_after_validate(state):
    failures = state.get("failures", [])
    if not failures or state.get("gaps"):
        return "review"
    owner = failures[0]["owner"]
    return (
        "review"
        if state.get("retries", {}).get(owner, 0) >= MAX_RETRIES
        else owner
    )


@contextmanager
def build_graph(checkpoint_path, services):
    g = StateGraph(CarState)
    for name, fn in make_nodes(services).items():
        g.add_node(name, fn)
    g.add_node("validate", validate_node)
    order = [
        "research",
        "problem",
        "root_cause",
        "actions",
        "verify",
        "assemble",
        "validate",
    ]
    g.add_edge(START, "research")
    for first, second in zip(order, order[1:]):
        g.add_edge(first, second)
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            name: name
            for name in [
                "problem",
                "root_cause",
                "actions",
                "verify",
                "review",
            ]
        },
    )
    g.add_conditional_edges(
        "review",
        lambda s: "research" if s["status"] == "revise" else END,
        {"research": "research", END: END},
    )
    with SqliteSaver.from_conn_string(
        str(checkpoint_path)
    ) as saver:
        yield g.compile(checkpointer=saver)
