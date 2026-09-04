# Chapter 15 — 15.2 Streaming and Multi-Turn
from sqm_ai.llm import MODELS, client


class Conversation:
    """Keep the transcript; the API keeps nothing."""

    def __init__(
        self, system: str,
        model: str = MODELS["standard"],
    ):
        self.system = system
        self.model = model
        self.messages: list[dict] = []

    def send(self, user_text: str) -> str:
        self.messages.append(
            {"role": "user", "content": user_text}
        )
        response = client.messages.create(
            model=self.model,
            max_tokens=1_024,
            system=self.system,
            messages=self.messages,
        )
        reply = next(
            b.text for b in response.content if b.type == "text"
        )
        self.messages.append(
            {"role": "assistant", "content": reply}
        )
        return reply
