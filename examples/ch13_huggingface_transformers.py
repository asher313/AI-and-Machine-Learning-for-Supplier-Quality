# Chapter 13 — 13.2 HuggingFace Transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=4,        # Northlake's NCR categories
)

inputs = tokenizer(
    ["Hole position out of tolerance by 2 mm.",
     "Minor scratch on non-critical surface."],
    padding=True,        # pad to the longest in the batch
    truncation=True,     # cut anything past max_length
    max_length=512,
    return_tensors="pt",
)
# keys: input_ids, attention_mask

outputs = model(**inputs)
print(outputs.logits.shape)      # (2, 4)
