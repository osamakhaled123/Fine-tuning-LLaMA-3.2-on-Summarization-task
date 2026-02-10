from huggingface_hub import login
from _config import token_name
import torch
from unsloth import FastLanguageModel
from peft import LoraConfig
from trl.trainer import SFTTrainer, SFTConfig
from transformers import (BitsAndBytesConfig,
                          TrainerCallback)
from datasets import load_dataset

login(token=token_name)

tokenized_train_set = load_dataset("osamakhaledML9/cnn_tokenized_datasets", split="train[:5%]")
tokenized_test_set = load_dataset("osamakhaledML9/cnn_tokenized_datasets", split="test")

model_name = "meta-llama/Llama-3.2-1B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    bias="none",
    lora_dropout=0.0,
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=1024,
    dtype=None,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=lora_config.r,
    target_modules=lora_config.target_modules,
    lora_alpha=lora_config.lora_alpha,
    lora_dropout=lora_config.lora_dropout,
    bias=lora_config.bias,
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.print_trainable_parameters()
class ClearMemoryCallback(TrainerCallback):
    def on_step_end(self, *args, **kwargs):
        torch.cuda.empty_cache()


training_args = SFTConfig(
    output_dir="results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    optim="paged_adamw_8bit",
    num_train_epochs=2,
    report_to="none",
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    push_to_hub=True,
    prediction_loss_only=True,
    hub_model_id="osamakhaledML9/llama-3.2-3b-cnn-dailymail-sft"
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=training_args,
    train_dataset=tokenized_train_set,
    eval_dataset=tokenized_test_set,
    compute_metrics=None,
    callbacks=[ClearMemoryCallback()]
)

trainer.train()