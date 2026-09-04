# Chapter 19 — 19.6 CrewAI
# examples/agents/crewai_ncr.py  (snapshot; see 19.4 note)
from crewai import Agent, Crew, Process, Task
from crewai.tools import tool
from langchain_anthropic import ChatAnthropic

from sqm_ai.llm import MODELS

llm = ChatAnthropic(model=MODELS["standard"])


@tool("Get supplier history")
def history(supplier_id: str) -> str:
    """Supplier nonconformance history, last 90 days."""
    return f"{supplier_id}: 3 dimensional NCRs in 90 days."


classifier = Agent(
    role="NCR Classifier",
    goal="Classify nonconformances by category and severity",
    backstory="A supplier-quality engineer who knows AS9100.",
    llm=llm,
)
investigator = Agent(
    role="Supplier History Investigator",
    goal="Put each NCR in the context of the supplier's record",
    backstory="An analyst who watches supplier trends.",
    tools=[history],
    llm=llm,
)
car_writer = Agent(
    role="Corrective Action Specialist",
    goal="Write clear, actionable corrective action requests",
    backstory="A process-improvement engineer.",
    llm=llm,
)

classify_task = Task(
    description="Classify this NCR: {description}",
    expected_output="Category and severity 1-5.",
    agent=classifier,
)
investigate_task = Task(
    description="Look up {supplier_id}; propose a disposition.",
    expected_output="A disposition with its rationale.",
    agent=investigator,
    context=[classify_task],
)
car_task = Task(
    description="If severity >= 3, draft a CAR for {ncr_id}.",
    expected_output="A CAR draft, or a statement of no need.",
    agent=car_writer,
    context=[classify_task, investigate_task],
)

crew = Crew(
    agents=[classifier, investigator, car_writer],
    tasks=[classify_task, investigate_task, car_task],
    process=Process.sequential,
)

result = crew.kickoff(inputs={
    "description": "Hole position 2 mm out, 12 units.",
    "supplier_id": "S-0417",
    "ncr_id": "NCR-2026-0042",
})
