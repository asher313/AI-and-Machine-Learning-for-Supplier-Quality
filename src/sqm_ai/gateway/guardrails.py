"""Heuristic signals supplement trusted classification; no claim of complete DLP."""

import re

from sqm_ai.gateway.detectors import (
    MarkingDetector,
    PIIDetector,
    injection_score,
    required_enclave,
)
from sqm_ai.gateway.policy import ENCLAVE_RANK, PolicyViolation

CITATION = re.compile(r"\[(\d+)\]")


class Guardrails:
    def __init__(self, policy):
        self.policy = policy
        self.pii, self.marks = PIIDetector(), MarkingDetector()

    def pre_check(
        self, payload, user, enclave, *, classification="unknown"
    ):
        # All labels/permissions must come from the authenticated application, not prompt text.
        if (
            classification not in ENCLAVE_RANK
            or enclave not in ENCLAVE_RANK
        ):
            raise PolicyViolation(["GW-CLASS"])
        if classification not in user.allowed_levels:
            raise PolicyViolation(["GW-AUTH"])
        codes, warnings = [], []
        hits = self.pii.scan(payload)
        if hits and "pii_in_prompts" not in user.permissions:
            codes.append("GW-PII")
        elif hits:
            warnings.append("authorized personal-data processing")
        need = required_enclave(
            [classification, *self.marks.scan(payload)]
        )
        if (
            ENCLAVE_RANK[enclave] < ENCLAVE_RANK[need]
            or ENCLAVE_RANK[classification] < ENCLAVE_RANK[need]
        ):
            codes.append("GW-MARK")
        if any(
            re.search(p, payload, re.IGNORECASE)
            for p in self.policy.forbidden_patterns
        ):
            codes.append("GW-PATTERN")
        score = injection_score(payload)
        if score >= self.policy.injection_threshold:
            codes.append("GW-INJECT")
        elif score:
            warnings.append(
                "injection phrase signal; not a calibrated probability"
            )
        if codes:
            raise PolicyViolation(codes)
        return warnings

    def post_check(self, response, sources, *, enclave):
        warnings = []
        if (
            self.marks.scan(response)
            and ENCLAVE_RANK[
                required_enclave(self.marks.scan(response))
            ]
            > ENCLAVE_RANK[enclave]
        ):
            raise PolicyViolation(["GW-POST-MARK"])
        if any(
            re.search(p, response, re.IGNORECASE)
            for p in self.policy.forbidden_patterns
        ):
            raise PolicyViolation(["GW-POST-PATTERN"])
        if self.pii.scan(response):
            response = self.pii.redact(response)
            warnings.append(
                "personal-data pattern redacted; residual identifiers may remain"
            )
        if self.policy.validate_citations:
            nums = [int(n) for n in CITATION.findall(response)]
            if (
                not sources
                or not nums
                or any(n < 1 or n > len(sources) for n in nums)
            ):
                raise PolicyViolation(["GW-CITE"])
        return response, warnings
