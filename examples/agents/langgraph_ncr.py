# Chapter 19 — 19.4 LangGraph
# examples/agents/langgraph_ncr.py  (snapshot; see note
# above. parse_severity(reply) is a four-line helper that
# reads the severity out of the classifier's reply and
# returns 0 when it cannot find one.)
import operator
from typing import Annotated, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from sqm_ai.llm import MODELS


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # accumulates
    ncr_id: str
    supplier_id: str
    severity: int


@tool
def get_supplier_history(supplier_id: str) -> str:
    """Supplier nonconformance history, last 90 days."""
    return f"{supplier_id}: 3 dimensional NCRs in 90 days."


@tool
def draft_car(
    ncr_id: str, supplier_id: str, root_cause: str
) -> str:
    """Open a corrective action request draft."""
    return f"CAR draft opened for {ncr_id}."


tools = [get_supplier_history, draft_car]
llm = ChatAnthropic(model=MODELS["standard"]).bind_tools(tools)


def classify_node(state: AgentState) -> dict:
    reply = llm.invoke(state["messages"])
    return {
        "messages": [reply],
        "severity": parse_severity(reply),
    }


def investigate_node(state: AgentState) -> dict:
    return {"messages": [llm.invoke(state["messages"])]}


def tool_node(state: AgentState) -> dict:
    by_name = {t.name: t for t in tools}
    out = []
    for call in state["messages"][-1].tool_calls:
        result = by_name[call["name"]].invoke(call["args"])
        out.append(ToolMessage(
            content=str(result), tool_call_id=call["id"],
        ))
    return {"messages": out}


def car_node(state: AgentState) -> dict:
    nudge = HumanMessage(
        content="Severity is 3 or more. Draft the CAR."
    )
    return {"messages": [llm.invoke(state["messages"] + [nudge])]}


def route(state: AgentState) -> str:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    if state["severity"] >= 3:
        return "car"
    return END


graph = StateGraph(AgentState)
graph.add_node("classify", classify_node)
graph.add_node("investigate", investigate_node)
graph.add_node("tools", tool_node)
graph.add_node("car", car_node)

graph.set_entry_point("classify")
graph.add_edge("classify", "investigate")
graph.add_conditional_edges(
    "investigate", route,
    {"tools": "tools", "car": "car", END: END},
)
graph.add_edge("tools", "investigate")
graph.add_edge("car", END)

app = graph.compile()

result = app.invoke({
    "messages": [HumanMessage(content="Process NCR-2026-0042")],
    "ncr_id": "NCR-2026-0042",
    "supplier_id": "S-0417",
    "severity": 0,
})
