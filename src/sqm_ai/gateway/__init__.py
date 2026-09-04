# sqm_ai/gateway/__init__.py
import time
import uuid
from dataclasses import dataclass

from sqm_ai.gateway.audit import AuditWriter, sha256
from sqm_ai.gateway.guardrails import Guardrails
from sqm_ai.gateway.policy import (
    BudgetExceeded, GatewayUser, Policy, PolicyViolation,
)
from sqm_ai.gateway.router import (
    BudgetLedger, ModelRouter, cost_usd,
)

REFUSAL = "This request cannot be processed. Reference: {codes}"


@dataclass
class GatewayResult:
    text: str | None
    blocked: bool
    codes: list[str]
    warnings: list[str]
    trace_id: str


class Gateway:
    def __init__(self, policy: Policy, dsn: str):
        self.policy = policy
        self.guard = Guardrails(policy)
        self.audit = AuditWriter(dsn)
        self.ledger = BudgetLedger(
            dsn, policy.monthly_budget_usd,
            policy.soft_alert_fraction,
        )
        self.router = ModelRouter()

    def complete(
        self, *, prompt: str, system: str, user: GatewayUser,
        tool_name: str, enclave: str, tier: str = "standard",
        essential: bool = False, sources: list[str] | None = None,
        max_tokens: int = 1024,
    ) -> GatewayResult:
        trace_id = str(uuid.uuid4())
        row = {
            "trace_id": trace_id, "user_id": user.user_id,
            "tool_name": tool_name, "enclave": enclave,
            "model": None, "prompt_hash": sha256(prompt),
            "prompt_tokens": None, "response_hash": None,
            "response_tokens": None, "pre_warnings": [],
            "post_warnings": [], "blocked": False,
            "block_codes": [], "latency_ms": None, "cost_usd": 0,
        }
        t0 = time.perf_counter()
        try:
            row["pre_warnings"] = self.guard.pre_check(
                prompt, user, enclave
            )
            budget = self.ledger.state()
            if budget == "soft":
                row["pre_warnings"].append("budget soft alert")
            client, model = self.router.choose(
                enclave, tier, budget, essential
            )
        except (PolicyViolation, BudgetExceeded) as exc:
            codes = getattr(exc, "codes", ["GW-BUDG"])
            row.update(blocked=True, block_codes=codes)
            self.audit.write(row)
            return GatewayResult(
                REFUSAL.format(codes=",".join(codes)),
                True, codes, [], trace_id,
            )

        msg = client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(
            b.text for b in msg.content if b.type == "text"
        )
        text, post = self.guard.post_check(text, sources)
        row.update(
            model=model, prompt_tokens=msg.usage.input_tokens,
            response_hash=sha256(text),
            response_tokens=msg.usage.output_tokens,
            post_warnings=post,
            latency_ms=int((time.perf_counter() - t0) * 1000),
            cost_usd=cost_usd(
                model, msg.usage.input_tokens,
                msg.usage.output_tokens,
            ),
        )
        self.audit.write(row)
        return GatewayResult(
            text, False, [], row["pre_warnings"] + post, trace_id
        )
