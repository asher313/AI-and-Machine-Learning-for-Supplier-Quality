# Chapter 19: one guarded loop, four orchestration examples

The chapter compares orchestration while holding the NCR, tool policies, and
scripted model responses constant. Generate the fixture; do not commit the
resulting `data/agents/replay.json`:

```bash
python scripts/generate_agent_data.py
python examples/agents/plain_ncr.py
python examples/agents/langgraph_ncr.py
python examples/agents/autogen_ncr.py
python examples/agents/crewai_ncr.py
```

Each example accepts `--input` for a generated fixture at another path. All
produce the same scripted suggestion in four model turns and three tools with
zero API requests. The three-row history is a deliberately small protocol
fixture, not Cobalt's canonical monthly/90-day history or empirical evidence.
The tool query used with an approved database reports the full 90-day total
and at most ten examples, excludes the current NCR, and respects an as-of
upper boundary. A caller must supply a read-only connection authorized for the
source data and the trusted NCR/supplier scope.

The examples make proposals only. They cannot approve a disposition, create
an operational CAR, send a message, or release product. CAR proposals require
triage, history, and severity >=3 in executable policy. Revising triage
invalidates an existing CAR proposal. The content hash is a repeatable proposal
identifier, not durable write idempotency. The model's final prose remains a
suggestion whose factual quality is not tested by this replay.

## Framework environment

Use a separate environment to isolate the optional framework dependency set:

```bash
uv venv .venv-agents --python 3.12
uv pip install --python .venv-agents/bin/python \
  -r requirements-agents-lock.txt
uv pip install --python .venv-agents/bin/python --no-deps -e .
```

Run the commands above with that environment's Python. Exact top-level pins
are recorded in `requirements-agents.txt`; the complete tested environment is
in `requirements-agents-lock.txt`. The normal model-training environment does
not need these frameworks. On Windows use `.venv-agents/Scripts/python.exe`.

- Plain Python executes the shared loop directly.
- LangGraph 1.2.11 wraps a step in state transitions. Its default CLI is
  in-memory; the test closes a SQLite checkpoint connection, reconstructs a
  graph, and resumes the persisted thread. Production restarts, external
  effects, and approval UI need further integration tests.
- AutoGen Core 0.7.5 uses explicit runtime messages; it does not use a model
  for speaker selection. AutoGen is maintenance-only; Microsoft recommends
  Agent Framework for new projects. This example explains existing systems,
  rather than recommending a new AutoGen deployment.
- CrewAI 1.15.21 uses a Python Flow router. CrewAI also supports role/task crews
  and conditional tasks, so mandatory branching need not live in a prompt.
  Optional telemetry is disabled in this teaching entry point. This version
  may initialize its local application-support directory during import.

A turn cap limits turns, not every HTTP request or tool call. The shared
executor separately limits tool calls, rejects oversized batches before
execution, returns safe error results, and stops on truncated output. The SDK
adapter has its own bounded retries and token budget. For production, add
end-to-end deadlines, protected run storage, access checks on resume, and
idempotent transaction boundaries where writes are allowed.

Validation: synthetic protocol replay through all four real runtimes; tool
scope and CAR policy tests; complete SDK/tool-result pairing through a mocked
HTTP transport; a persisted LangGraph resume; and an isolated PostgreSQL
history-count/date test. No live model-quality, cost, latency, or comparative
framework benchmark was run. The book's historical bake-off table is fictional.

Primary references checked September 11, 2026:

- [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence)
- [LangGraph interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts)
- [AutoGen support status](https://github.com/microsoft/autogen)
- [AutoGen Core messages](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/framework/message-and-communication.html)
- [CrewAI Flows](https://docs.crewai.com/en/concepts/flows)
- [CrewAI conditional tasks](https://docs.crewai.com/en/learn/conditional-tasks)
