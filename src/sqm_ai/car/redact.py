# src/sqm_ai/car/redact.py
import re
from dataclasses import dataclass


@dataclass
class Redactor:
    """Two-way map between real names and prompt tokens."""

    forward: dict[str, str]        # real -> token
    backward: dict[str, str]       # token -> real

    @classmethod
    def build(cls, names: dict[str, str]) -> "Redactor":
        """names: supplier_id -> supplier_name."""
        fwd = {}
        for i, (sid, name) in enumerate(sorted(names.items())):
            token = f"SUPPLIER_{chr(ord('A') + i)}"
            fwd[name] = token
            fwd[sid] = token
        return cls(fwd, {v: k for k, v in fwd.items()})

    def out(self, text: str) -> str:
        """Real world -> model. Longest names first."""
        for real in sorted(self.forward, key=len, reverse=True):
            text = text.replace(real, self.forward[real])
        return text

    def back(self, text: str, ids: dict[str, str]) -> str:
        """Model -> real world, using the id map."""
        for token, real in self.backward.items():
            text = text.replace(token, ids.get(token, real))
        return text

    def leaked(self, text: str) -> list[str]:
        """Real names that survived into model-bound text."""
        return [
            real for real in self.forward
            if re.search(rf"\b{re.escape(real)}\b", text)
        ]
