# Chapter 13 — 13.3 LoRA and QLoRA
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    dtype=torch.float32,  # simple CPU-compatible starting point
    # Trainer owns placement; use an explicit supported GPU dtype there
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                 # rank: low = fewer parameters
    lora_alpha=32,        # scaling, usually 2x the rank
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
