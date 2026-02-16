import torch
import numpy as np
import evaluate
import tqdm


def build_prompt(article):
    return f"Summarize the following article:\n{article}\nSummary: "


@torch.inference_mode()
def summarize_beam_search(article, model, tokenizer, max_tokens=128):
    prompt = build_prompt(article)
    inputs = tokenizer(prompt,
                       return_tensors="pt",
                       truncation=True,
                       max_length=1024).to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=3 if torch.cuda.is_available() else 1,
            early_stopping=True,
            no_repeat_ngram_size=2,
            length_penalty=1.0,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs['input_ids'].shape[1]:]

    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated_text

def bert_score(tokenized_set, model, tokenizer, num_examples=10):
    predictions = []
    references = []

    for example in tqdm.tqdm(tokenized_set.select(range(num_examples))):
        article = tokenizer.decode(example["input_ids"][:example["labels"].index(-100)])

        summary = summarize_beam_search(article, model, tokenizer)

        reference = tokenizer.decode(
            [token for token in example['labels'] if token != -100],
            skip_special_tokens=True)

        predictions.append(summary)
        references.append(reference)

    bertscore = evaluate.load("bertscore")

    results = bertscore.compute(predictions=predictions,
                                references=references,
                                lang="en")

    return {
        "precision": np.mean(results["precision"]),
        "recall": np.mean(results["recall"]),
        "f1": np.mean(results["f1"]),
    }
