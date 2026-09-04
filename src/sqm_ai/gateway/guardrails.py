# sqm_ai/gateway/guardrails.py
import re

from sqm_ai.gateway.detectors import (
    MarkingDetector, PIIDetector, injection_score,
    required_enclave,
)
from sqm_ai.gateway.policy import (
    ENCLAVE_RANK, GatewayUser, Policy, PolicyViolation,
)

CITATION = re.compile(r"\[(\d+)\]")


class Guardrails:
    def __init__(self, policy: Policy):
        self.policy = policy
        self.pii = PIIDetector()
        self.marks = MarkingDetector()

    def pre_check(
        self, prompt: str, user: GatewayUser, enclave: str
    ) -> list[str]:
        """Return warnings; raise PolicyViolation to block."""
        codes, reasons, warnings = [], [], []

        # 1. personal data — allowed only with a permission
        hits = self.pii.scan(prompt)
        if hits and "pii_in_prompts" not in user.permissions:
            codes.append("GW-PII")
            reasons.append(f"personal data detected: {hits}")
        elif hits:
            warnings.append(f"pii allowed by permission: {hits}")

        # 2. markings — content must not outrank its enclave
        need = required_enclave(self.marks.scan(prompt))
        if ENCLAVE_RANK[enclave] < ENCLAVE_RANK[need]:
            codes.append("GW-MARK")
            reasons.append(
                f"content requires {need}; running in {enclave}"
            )

        # 3. forbidden patterns (program and NDA names)
        for pat in self.policy.forbidden_patterns:
            if re.search(pat, prompt, re.IGNORECASE):
                codes.append("GW-PATTERN")
                reasons.append("matches a forbidden pattern")
                break

        # 4. injection heuristic
        score = injection_score(prompt)
        if score >= self.policy.injection_threshold:
            codes.append("GW-INJECT")
            reasons.append(f"suspected injection ({score:.2f})")
        elif score > 0:
            warnings.append(f"injection score {score:.2f}")

        if codes:
            raise PolicyViolation(codes, reasons)
        return warnings

    def post_check(
        self, response: str, sources: list[str] | None
    ) -> tuple[str, list[str]]:
        """Return (possibly redacted response, warnings)."""
        warnings = []

        if self.pii.scan(response):
            response = self.pii.redact(response)
            warnings.append("pii redacted from response")

        if self.marks.scan(response):
            warnings.append("response carries a marking")

        if self.policy.validate_citations and sources is not None:
            cited = {int(n) for n in CITATION.findall(response)}
            n_src = len(sources)
            bad = sorted(
                n for n in cited
                if n > n_src or n < 1
            )
            if bad:
                warnings.append(f"unsupported citations: {bad}")

        return response, warnings
