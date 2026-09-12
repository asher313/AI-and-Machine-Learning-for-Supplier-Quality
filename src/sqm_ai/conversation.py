"""Text conversation with transactional local history (no tool execution)."""

from sqm_ai.llm import (
    MODELS,
    client,
    count_tokens,
    request_options,
    text_response,
    with_retry,
)


class Conversation:
    def __init__(
        self, system, model=MODELS["standard"], window=200_000
    ):
        self.system, self.model, self.window = (
            system,
            model,
            window,
        )
        self.messages = []

    def send(self, user_text):
        pending = self.messages + [
            {"role": "user", "content": user_text}
        ]
        size = count_tokens(
            model=self.model, system=self.system, messages=pending
        )
        if size + 1024 + 512 > self.window:
            raise ValueError(
                "history exceeds budget; explicitly summarize or start a new conversation"
            )
        response = with_retry(
            client.messages.create,
            model=self.model,
            max_tokens=1024,
            system=self.system,
            messages=pending,
            **request_options(self.model),
        )
        reply = text_response(response)
        # Retain thinking/signature blocks required when resubmitting history.
        blocks = [
            block.model_dump(exclude_none=True)
            for block in response.content
        ]
        self.messages = pending + [
            {"role": "assistant", "content": blocks}
        ]
        return reply
