# src/sqm_ai/car/memory.py
from dataclasses import dataclass, field

from sqm_ai.assistant.access import AssistantUser
from sqm_ai.assistant.retrieve import fetch_chunks
from sqm_ai.llm import MODELS, client, with_retry

SUMMARIZE = (
    "Summarize this corrective-action session in under 80 "
    "words: the defect, the accepted root cause, the "
    "verification, and the outcome. No supplier names."
)


@dataclass
class CarMemory:
    """Three tiers: working, episodic, semantic."""

    supplier_id: str
    user: AssistantUser           # whose access applies
    working: list[str] = field(default_factory=list)
    episodic: list[str] = field(default_factory=list)
    max_working: int = 20

    def add(self, note: str) -> None:
        self.working.append(note)
        if len(self.working) > self.max_working:
            oldest = self.working[: self.max_working // 2]
            self.episodic.append(self._summarize(oldest))
            self.working = self.working[self.max_working // 2:]

    def _summarize(self, notes: list[str]) -> str:
        response = with_retry(
            client.messages.create,
            model=MODELS["fast"],
            max_tokens=200,
            system=SUMMARIZE,
            messages=[{
                "role": "user", "content": "\n".join(notes),
            }],
        )
        return "".join(
            b.text for b in response.content
            if b.type == "text"
        )

    def recall(self, query: str, k: int = 3) -> str:
        """Working in full, episodic recent, semantic top-k."""
        parts = ["\n".join(self.working)]
        parts += self.episodic[-3:]
        chunks = fetch_chunks(query, self.user, keep=k)
        parts += [c["content"] for c in chunks]
        return "\n\n".join(p for p in parts if p)
