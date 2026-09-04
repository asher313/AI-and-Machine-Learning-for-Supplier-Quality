# Chapter 13 — 13.3 LoRA and QLoRA
from peft import PeftModel
from transformers import AutoModelForCausalLM

model.save_pretrained("./lora_adapters")

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B"
)
model = PeftModel.from_pretrained(base, "./lora_adapters")
