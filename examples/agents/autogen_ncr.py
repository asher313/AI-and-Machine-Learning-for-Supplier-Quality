# Chapter 19 — 19.5 AutoGen
# examples/agents/autogen_ncr.py  (snapshot; see 19.4 note)
import os

import autogen

from sqm_ai.agent.tools import draft_car, get_supplier_history
from sqm_ai.llm import MODELS

config_list = [{
    "model": MODELS["standard"],
    "api_key": os.environ["ANTHROPIC_API_KEY"],
    "api_type": "anthropic",
}]
llm_config = {"config_list": config_list}

classifier = autogen.AssistantAgent(
    name="classifier",
    system_message=(
        "You classify NCRs. Reply with category (cosmetic, "
        "dimensional, material, functional) and severity 1-5."
    ),
    llm_config=llm_config,
)

investigator = autogen.AssistantAgent(
    name="investigator",
    system_message=(
        "You pull supplier history and propose a disposition."
    ),
    llm_config=llm_config,
)

car_writer = autogen.AssistantAgent(
    name="car_writer",
    system_message=(
        "You draft corrective action requests. Speak only "
        "when severity is 3 or higher."
    ),
    llm_config=llm_config,
)

proxy = autogen.UserProxyAgent(
    name="proxy",
    human_input_mode="NEVER",
    code_execution_config=False,
)


@investigator.register_for_llm(
    description="Supplier NCR history, last 90 days"
)
@proxy.register_for_execution()
def history_tool(supplier_id: str) -> str:
    return get_supplier_history(supplier_id)


chat = autogen.GroupChat(
    agents=[proxy, classifier, investigator, car_writer],
    messages=[],
    max_round=12,
    speaker_selection_method="auto",   # a model picks
)
manager = autogen.GroupChatManager(
    groupchat=chat, llm_config=llm_config,
)

proxy.initiate_chat(
    manager,
    message=(
        "Process NCR-2026-0042 from supplier S-0417: hole "
        "position 2 mm out of tolerance, 12 units."
    ),
)
