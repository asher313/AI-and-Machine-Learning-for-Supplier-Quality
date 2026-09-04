# src/sqm_ai/car/nodes.py
#
# Chapter 20.3's graph listing names these seven nodes and
# describes them: "Each is a thin wrapper: it reads the
# fields it needs from the state, calls its role's prompt
# (Section 20.4 shows the root-cause one in full), and
# returns the fields it produces."  Only the root-cause node
# is printed in full, in Section 20.4.
#
# TODO(book): condensed in Chapter 20 — the six remaining
# role nodes are described, not listed. Complete before
# production use.
from sqm_ai.car.state import CarState


def problem_node(state: CarState) -> dict:
    raise NotImplementedError("Chapter 20.3: not listed")


def research_node(state: CarState) -> dict:
    raise NotImplementedError("Chapter 20.3: not listed")


def root_cause_node(state: CarState) -> dict:
    """Section 20.4 lists the prompt and the schema this
    node calls; see sqm_ai.car.root_cause."""
    raise NotImplementedError("Chapter 20.4: wiring not listed")


def actions_node(state: CarState) -> dict:
    raise NotImplementedError("Chapter 20.3: not listed")


def verify_node(state: CarState) -> dict:
    raise NotImplementedError("Chapter 20.3: not listed")


def assemble_node(state: CarState) -> dict:
    raise NotImplementedError("Chapter 20.3: not listed")


def escalate_node(state: CarState) -> dict:
    raise NotImplementedError("Chapter 20.3: not listed")
