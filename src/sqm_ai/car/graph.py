# src/sqm_ai/car/graph.py
# TODO(book): condensed in Chapter 20 — complete before production use.
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

# The six role nodes and the escalation node live in
# sqm_ai/car/nodes.py. Each is a thin wrapper: it reads the
# fields it needs from the state, calls its role's prompt
# (Section 20.4 shows the root-cause one in full), and
# returns the fields it produces. validate_node is the
# listing below.
from sqm_ai.car.nodes import (
    actions_node, assemble_node, escalate_node, problem_node,
    research_node, root_cause_node, verify_node,
)
from sqm_ai.car.state import CarState
from sqm_ai.car.validate import validate_node

MAX_RETRIES = 2


def route_after_validate(state: CarState) -> str:
    """Send the draft back to whoever produced the problem."""
    if state.get("escalation"):
        return "escalate"
    failures = state.get("failures") or []
    if not failures:
        return END
    owner = failures[0].split(":")[1]          # FAIL:node:why
    if state["retries"].get(owner, 0) >= MAX_RETRIES:
        return "escalate"
    return owner


def build_graph(checkpoint_path: str):
    g = StateGraph(CarState)
    g.add_node("research", research_node)
    g.add_node("problem", problem_node)
    g.add_node("root_cause", root_cause_node)
    g.add_node("actions", actions_node)
    g.add_node("verify", verify_node)
    g.add_node("assemble", assemble_node)
    g.add_node("validate", validate_node)
    g.add_node("escalate", escalate_node)

    g.set_entry_point("research")
    g.add_edge("research", "problem")
    g.add_edge("problem", "root_cause")
    g.add_edge("root_cause", "actions")
    g.add_edge("actions", "verify")
    g.add_edge("verify", "assemble")
    g.add_edge("assemble", "validate")
    g.add_conditional_edges(
        "validate", route_after_validate,
        {
            "problem": "problem",
            "root_cause": "root_cause",
            "actions": "actions",
            "verify": "verify",
            "escalate": "escalate",
            END: END,
        },
    )
    g.add_edge("escalate", END)
    return g.compile(
        checkpointer=SqliteSaver.from_conn_string(
            checkpoint_path
        ),
        interrupt_before=["escalate"],
    )
