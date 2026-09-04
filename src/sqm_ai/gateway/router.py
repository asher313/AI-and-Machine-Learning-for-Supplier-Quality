# sqm_ai/gateway/router.py
import os

import psycopg
from anthropic import AnthropicBedrockMantle

from sqm_ai.llm import MODELS, client as commercial
from sqm_ai.gateway.policy import BudgetExceeded

# USD per million tokens (input, output); first-party list
# prices as of September 2026 — mirror of the Ch 15 table.
PRICE_PER_MTOK = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

MTD_SQL = """
SELECT COALESCE(SUM(cost_usd), 0) FROM llm_audit
WHERE ts >= date_trunc('month', now())
"""


def cost_usd(model: str, in_tok: int, out_tok: int) -> float:
    base = model.removeprefix("anthropic.")
    p_in, p_out = PRICE_PER_MTOK[base]
    return (in_tok * p_in + out_tok * p_out) / 1_000_000


class BudgetLedger:
    def __init__(self, dsn: str, cap: float, soft: float):
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.cap, self.soft = cap, soft

    def month_to_date(self) -> float:
        return float(self.conn.execute(MTD_SQL).fetchone()[0])

    def state(self) -> str:
        spent = self.month_to_date()
        if spent >= self.cap:
            return "hard"
        if spent >= self.soft * self.cap:
            return "soft"
        return "ok"


class ModelRouter:
    """controlled -> Bedrock in the enclave; else commercial."""

    def __init__(self) -> None:
        self.enclave = AnthropicBedrockMantle(
            aws_region=os.environ["NL_ENCLAVE_REGION"]
        )

    def choose(self, enclave: str, tier: str, budget: str,
               essential: bool):
        if budget == "hard":
            if not essential:
                raise BudgetExceeded("monthly cap reached")
            tier = "fast"                  # degrade, don't stop
        base = MODELS[tier]
        if enclave == "controlled":
            return self.enclave, "anthropic." + base
        return commercial, base
