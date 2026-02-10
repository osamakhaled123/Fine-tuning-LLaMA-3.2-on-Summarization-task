import torch
from huggingface_hub import login
from _config import token_name
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
import inference

login(token=token_name)

BASE_MODEL="meta-llama/Llama-3.2-3B"
ADAPTER_REPO="osamakhaledML9/llama-3.2-3b-cnn-dailymail-sft"

tokenizer=AutoTokenizer.from_pretrained(ADAPTER_REPO)

model=AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto"
)

model=PeftModel.from_pretrained(model, ADAPTER_REPO)
model.config.use_cache=True
model.eval()

tokenized_set=load_dataset("osamakhaledML9/cnn_tokenized_datasets", split="test")

result = inference.bert_score(tokenized_set=tokenized_set,
                              model=model,
                              tokenizer=tokenizer,
                              num_examples=10)

print(result)