import os
import json
import tqdm
import datasets
from huggingface_hub import HfApi
from _config import token_name


train_data = datasets.load_from_disk("./data/cnn_train_set")
test_data = datasets.load_from_disk("./data/cnn_test_set")


OUTPUT_DIR = ".\jsonl_shards"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def sharding(dataset, set, SHARD_SIZE=10_000):
    buffer = []
    shard_idx = 0

    for i, example in enumerate(tqdm.tqdm(dataset)):
        buffer.append(example)

        if len(buffer) == SHARD_SIZE:
            shard_path = os.path.join(
                OUTPUT_DIR, f"{set}_{shard_idx:04d}.jsonl"
            )

            with open(shard_path, "w", encoding="utf-8") as f:
                for row in buffer:
                    f.write(json.dumps(row) + "\n")

            buffer.clear()
            shard_idx += 1

    if buffer:
        shard_path = os.path.join(
            OUTPUT_DIR, f"{set}_{shard_idx:04d}.jsonl"
        )

        with open(shard_path, "w", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row) + "\n")


sharding(train_data, set="train")
sharding(test_data, set="test")

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_HUB_DISABLE_XET"] = "1"


api = HfApi()

api.upload_folder(
    repo_id="osamakhaledML9/cnn_tokenized_datasets",
    folder_path="./jsonl_shards/",
    repo_type="dataset",
    path_in_repo="",
    commit_message="Updating Shards of datasets enabling hf_transfer",
    token=token_name
)
