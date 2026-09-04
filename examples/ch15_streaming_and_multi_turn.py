# Chapter 15 — 15.2 Streaming and Multi-Turn
from sqm_ai.llm import MODELS, client

with client.messages.stream(
    model=MODELS["frontier"],
    max_tokens=8_000,
    system="You draft corrective action requests.",
    messages=[{"role": "user", "content": car_request}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
    final = stream.get_final_message()

print()
print(final.stop_reason, final.usage.output_tokens)
