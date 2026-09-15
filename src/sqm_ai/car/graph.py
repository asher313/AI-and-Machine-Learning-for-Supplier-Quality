"""Bounded repairs plus a durable human decision on every finished draft."""

from contextlib import contextmanager, closing
from pathlib import Path
import sqlite3

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from sqm_ai.car.nodes import make_nodes
from sqm_ai.car.state import (
    CarState,
    ReviewDecision,
    draft_revision,
)
from sqm_ai.car.validate import validate_node

MAX_RETRIES = 2  # Two additional repair attempts after the original role call.


class ScopedGraph:
    """Small local API: validate scope before disclosure; serialize checkpoint access.

    A separate SQLite lock DB holds an exclusive transaction across graph calls.
    It works across processes on a local filesystem. It is not a distributed
    checkpoint service; production storage needs its own transaction design.
    """

    def __init__(self, graph, services, checkpoint_path):
        self._graph, self._services = graph, services
        self._lock_path = (
            str(checkpoint_path) + ".scope-lock.sqlite"
        )

    @contextmanager
    def _locked(self):
        with closing(
            sqlite3.connect(self._lock_path, timeout=30)
        ) as conn:
            with conn:
                conn.execute("BEGIN EXCLUSIVE")
                yield

    def _state(self, config):
        if not config.get("configurable", {}).get("thread_id"):
            raise ValueError("stable per-run thread ID required")
        if set(config["configurable"]) != {"thread_id"}:
            raise ValueError(
                "historical checkpoint overrides are not supported"
            )
        state = self._graph.get_state(config)
        if state.values:
            scope = self._services.scope()
            if (
                state.values.get("run_scope") != scope
                or state.values.get("ncr_id") != scope["ncr_id"]
                or state.values.get("supplier_id")
                != scope["supplier_id"]
            ):
                raise PermissionError(
                    "checkpoint scope differs from authorized run"
                )
        return state

    def get_state(self, config):
        with self._locked():
            return self._state(config)

    def invoke(self, request, config):
        from langgraph.types import Command

        with self._locked():
            old = self._state(config)
            if isinstance(request, Command):
                if not old.values or old.next != ("review",):
                    raise ValueError(
                        "no paused review for this thread"
                    )
                if (
                    request.update
                    or request.goto
                    or request.graph
                ):
                    raise ValueError(
                        "only an explicit review resume is supported"
                    )
                # Reject bad resumes before LangGraph persists interrupt values.
                decision = ReviewDecision.model_validate(
                    request.resume
                )
                if decision.draft_revision != draft_revision(
                    old.values
                ):
                    raise ValueError(
                        "stale draft revision; review the current draft"
                    )
                if not self._services.reviewer_authorized(
                    decision
                ):
                    raise PermissionError(
                        "reviewer not authorized"
                    )
                if decision.evidence:
                    self._services.check_scope(decision.evidence)
                if decision.action == "accept_draft" and (
                    old.values.get("failures")
                    or old.values.get("gaps")
                ):
                    raise ValueError(
                        "resolve failures and explicit gaps before accepting this draft"
                    )
            else:
                if old.values:
                    raise ValueError(
                        "thread exists; resume its current review"
                    )
                scope = self._services.scope()
                if (
                    not isinstance(request, dict)
                    or set(request) - {"ncr_id", "supplier_id"}
                    or request.get("ncr_id") != scope["ncr_id"]
                    or request.get(
                        "supplier_id", scope["supplier_id"]
                    )
                    != scope["supplier_id"]
                ):
                    raise PermissionError(
                        "initial request differs from authorized scope"
                    )
                request = {
                    "ncr_id": scope["ncr_id"],
                    "supplier_id": scope["supplier_id"],
                    "run_scope": scope,
                }
            return self._graph.invoke(request, config)


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
    if str(checkpoint_path) == ":memory:":
        raise ValueError("use a persistent local checkpoint path")
    Path(checkpoint_path).parent.mkdir(
        parents=True, exist_ok=True
    )
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
        yield ScopedGraph(
            g.compile(checkpointer=saver),
            services,
            checkpoint_path,
        )
