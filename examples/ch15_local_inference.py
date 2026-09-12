# Chapter 15 teaching listing. Supply the inputs described in the text.
import ollama

# SYSTEM and NCRClassification: imports as in §15.3.
response = ollama.chat(
    model="llama3.2:3b",   # pull this explicit small-model tag first
    messages=[
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": description},
    ],
    options={"temperature": 0, "num_predict": 400},
    format=NCRClassification.model_json_schema(),   # schema-bound
)
result = NCRClassification.model_validate_json(
    response["message"]["content"]
)
