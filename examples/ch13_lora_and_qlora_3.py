# Chapter 13 — 13.3 LoRA and QLoRA
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # 'nf4' or 'fp4'
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,  # extra compression
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.3-70B",
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)
# then attach LoRA exactly as above
