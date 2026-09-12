"""CrewAI Flow with a Python route around the shared bounded agent step."""

import argparse
import os

# Disable optional framework telemetry before importing its runtime.
os.environ.setdefault("CREWAI_TELEMETRY_DISABLED", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field

from sqm_ai.agent.replay import load_replay, restore, snapshot


class State(BaseModel):
    run: dict = Field(default_factory=dict)
    route: str = "continue"


def execute(run, call):
    class NCRFlow(Flow[State]):
        @start()
        def begin(self):
            self.state.run = snapshot(run)

        @listen(or_(begin, "again"))
        def step(self):
            current = restore(self.state.run, run.tools)
            self.state.route = current.advance(call)
            self.state.run = snapshot(current)

        @router(step)
        def choose(self):
            return (
                "done" if self.state.route == "done" else "again"
            )

        @listen("done")
        def finish(self):
            return self.state.run

    flow = NCRFlow(tracing=False)
    return flow.kickoff()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/agents/replay.json"
    )
    args = parser.parse_args()
    run, call = load_replay(args.input)
    print(execute(run, call)["answer"])
