# Chapter 13 — 13.3 LoRA and QLoRA
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import prepare_model_for_kbit_training

if not torch.cuda.is_available():
    raise RuntimeError("this QLoRA recipe targets a compatible CUDA GPU")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",       # 'nf4' or 'fp4'
    bnb_4bit_compute_dtype=(
        torch.bfloat16 if torch.cuda.is_bf16_supported()
        else torch.float16
    ),
    bnb_4bit_use_double_quant=True,  # extra compression
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B",  # small worked example
    quantization_config=bnb_config,
    device_map={"": 0},  # explicit single CUDA device for this recipe
)
model = prepare_model_for_kbit_training(model)
# then attach LoRA exactly as above
