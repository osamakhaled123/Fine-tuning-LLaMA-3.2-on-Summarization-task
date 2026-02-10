from huggingface_hub import login
from _config import token_name
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

login(token=token_name)

model_name = "meta-llama/Llama-3.2-1B"

train_data = load_dataset("cnn_dailymail", "3.0.0", split="train[:40%]")
test_data = load_dataset("cnn_dailymail", "3.0.0", split="test[:40%]")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def format_data(batch):
    input_ids = []
    attention_mask = []
    targets = []

    for article, highlights in zip(batch['article'], batch['highlights']):
        prompt = f"Summarize the following article:\n{article}\nSummary: "
        full_text = f"{prompt} {highlights}"

        tokenized_batch = tokenizer(full_text, truncation=True)
        tokenized_prompt_length = len(tokenizer(prompt, truncation=True)['input_ids'])

        tokenized_summary = tokenized_batch['input_ids'][tokenized_prompt_length:].copy()
        labels = [-100] * tokenized_prompt_length
        labels.extend(tokenized_summary)

        input_ids.append(tokenized_batch['input_ids'])
        attention_mask.append(tokenized_batch['attention_mask'])
        targets.append(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": targets
    }


tokenized_train_data = train_data.map(format_data, batched=True, remove_columns=train_data.column_names)
tokenized_test_data = test_data.map(format_data, batched=True, remove_columns=test_data.column_names)

dataset_dict = DatasetDict({"train":tokenized_train_data,
                            "test":tokenized_test_data})

dataset_dict.push_to_hub("osamakhaledML9/cnn_tokenized_datasets",
                         max_shard_size="500MB")