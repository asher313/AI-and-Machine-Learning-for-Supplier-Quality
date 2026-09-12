# 0007: orchestration for the teaching builds

Date: 2026-09-11. Status: teaching architecture decision, not company approval.

Use the shared bounded Python loop for short NCR proposal runs. Use LangGraph
for Build 5 where a persistent human checkpoint is a concrete requirement.
Keep validation, trusted scope, severity policy, and model adapters independent
of the framework. Test persisted resume and effects around retries separately.

AutoGen Core and CrewAI Flows pass the same scripted protocol replay. That
establishes software compatibility, not comparative answer quality. AutoGen
is maintenance-only and is retained to explain existing systems. CrewAI has
coded branching and persistence capabilities; it was not rejected for lacking
them. Revisit support status and deployment requirements before adoption and
at the next six-month architecture review.
