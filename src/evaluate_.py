import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import PeftModel
from datasets import load_dataset
import inference

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

BASE_MODEL="meta-llama/Llama-3.2-3B"
ADAPTER_REPO="osamakhaledML9/llama-3.2-3b-cnn-dailymail-sft"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,
                                          use_fast=True)
tokenizer.pad_token=tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,
                                             quantization_config=bnb_config,
                                             device_map="auto")

model = PeftModel.from_pretrained(model, ADAPTER_REPO)

model.config.use_cache = True
model.eval()

tokenized_set=load_dataset("osamakhaledML9/cnn_tokenized_datasets", split="test")

result = inference.bert_score(tokenized_set=tokenized_set,
                              model=model,
                              tokenizer=tokenizer,
                              num_examples=10)

print(result)