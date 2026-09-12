# Chapter 15 teaching listing. Supply the inputs described in the text.
from sqm_ai.llm import MODELS, client

response = client.messages.create(
    model=MODELS["standard"],
    max_tokens=512,
    system="You are an aerospace supplier-quality engineer.",
    messages=[
        {"role": "user",
         "content": "Classify this NCR: hole position 2 mm out."},
    ],
    output_config={"effort": "low"},   # short task; think less
)

text = next(b.text for b in response.content if b.type == "text")
print(text)
print(response.model, response.stop_reason)
print(response.usage.input_tokens, response.usage.output_tokens)
