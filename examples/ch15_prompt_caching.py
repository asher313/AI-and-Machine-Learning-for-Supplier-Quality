# Chapter 15 — 15.6 Prompt Caching
from sqm_ai.llm import MODELS, client

TAXONOMY = open("prompts/taxonomy.md").read()   # ~3,000 tokens

response = client.messages.create(
    model=MODELS["standard"],
    max_tokens=512,
    system=[
        {"type": "text", "text": SYSTEM},
        {
            "type": "text",
            "text": TAXONOMY,
            "cache_control": {"type": "ephemeral"},   # breakpoint
        },
    ],
    messages=[{"role": "user", "content": description}],
)
u = response.usage
print(u.cache_creation_input_tokens,   # written this call
      u.cache_read_input_tokens,       # served from cache
      u.input_tokens)                  # paid at full price
