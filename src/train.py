from huggingface_hub import login
from _config import token_name
import torch
import numpy as np
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, DataCollatorWithPadding
from datasets import load_dataset
import evaluate

login(token=token_name)

tokenized_train_set = load_dataset("osamakhaledML9/cnn_tokenized_datasets")
tokenized_test_set = load_dataset("osamakhaledML9/cnn_tokenized_datasets")

model_name = "meta-llama/Llama-3.2-3B"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    bias="none",
    lora_dropout=0.1,
    task_type="Causal_LM",
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
)

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             quantization_config=bnb_config,
                                             device_map="auto")
model.config.use_cache = False

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.add_eos_token
tokenizer.padding_side = "right"

data_collator = DataCollatorWithPadding(tokenizer=tokenizer,
                                        pad_to_multiple_of=8)

metric = evaluate.load("bertscore")


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    labels = np.where(labels != -100, labels, tokenizer.pad_token_type_id)

    decoded_logits = tokenizer.batch_decode(logits, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    results = metric.compute(predictions=decoded_logits,
                             references=decoded_labels,
                             lang="en"
                             )

    return {
        "precision_bert": np.mean(results['precision']),
        "recall_bert": np.mean(results['recall']),
        "f1_bert": np.mean(results['f1'])
    }


training_args = SFTConfig(
    output_dir="src/results",
    eval_strategy="step",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    report_to="none",
    bf16=True,
    logging_steps=10,
    save_steps=500,
    eval_steps=100
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    data_collator=data_collator,
    args=training_args,
    train_dataset=tokenized_test_set,
    eval_dataset=tokenized_test_set,
    compute_metrics=compute_metrics,

)

trainer.train()

trainer.push_to_hub()
