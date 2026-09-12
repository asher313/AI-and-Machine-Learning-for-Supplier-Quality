"""Run the shared synthetic NCR replay without an orchestration framework."""

import argparse

from sqm_ai.agent.loop import run_agent
from sqm_ai.agent.replay import load_replay

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/agents/replay.json"
    )
    args = parser.parse_args()
    run, call = load_replay(args.input)
    print(run_agent(run, call=call))
