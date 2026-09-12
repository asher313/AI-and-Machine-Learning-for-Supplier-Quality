"""AutoGen Core 0.7 replay: explicit runtime messages, no selector-model overhead."""

import argparse
import asyncio
from dataclasses import dataclass

from autogen_core import (
    AgentId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    message_handler,
)

from sqm_ai.agent.replay import load_replay, restore, snapshot


@dataclass
class Advance:
    state: dict


@dataclass
class Advanced:
    state: dict
    route: str


class NCRAgent(RoutedAgent):
    def __init__(self, scope, call):
        super().__init__("Bounded NCR proposal agent")
        self.scope, self.call = scope, call

    @message_handler
    async def advance(
        self, message: Advance, ctx: MessageContext
    ) -> Advanced:
        run = restore(message.state, self.scope)
        route = await asyncio.to_thread(run.advance, self.call)
        return Advanced(snapshot(run), route)


async def execute(run, call):
    runtime = SingleThreadedAgentRuntime()
    await NCRAgent.register(
        runtime, "ncr", lambda: NCRAgent(run.tools, call)
    )
    runtime.start()
    state = snapshot(run)
    try:
        for _ in range(run.max_steps):
            result = await runtime.send_message(
                Advance(state), AgentId("ncr", "teaching_run")
            )
            state = result.state
            if result.route == "done":
                return state
        raise RuntimeError("AutoGen host step budget exhausted")
    finally:
        await runtime.stop_when_idle()
        await runtime.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/agents/replay.json"
    )
    args = parser.parse_args()
    run, call = load_replay(args.input)
    print(asyncio.run(execute(run, call))["answer"])
