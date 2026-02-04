from huggingface_hub import login
from _config import token_name
import torch
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding

login(token=token_name)

model_name = "meta-llama/Llama-3.2-3B"

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.1,
    task_type="Causal_LM",
    target_modules=["q_proj", "v_proj",  "gate_proj", "up_proj", "down_proj"]
)

model = AutoModelForCausalLM.from_pretrained(model_name, , device_map="auto")
