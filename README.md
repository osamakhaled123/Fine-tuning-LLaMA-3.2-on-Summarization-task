# 🦙 LLaMA-3.2-3B CNN/DailyMail Summarization (QLoRA + Unsloth)

This project fine-tunes **LLaMA-3.2-3B** on the **CNN/DailyMail 3.0.0** dataset for abstractive text summarization using **Quantized LoRA (QLoRA)** with accelerated training via **Unsloth**.

The project demonstrates a complete LLM engineering pipeline:
- Efficient fine-tuning
- Evaluation with semantic metrics
- Optimized inference
- Deployment via Gradio on Hugging Face Spaces

🔗 **Live Demo:**  
https://huggingface.co/spaces/osamakhaledML9/LLaMA-3.0-3B-eng_summarizer_app

---

# 🚀 Project Overview

This project showcases:

- Parameter-Efficient Fine-Tuning (PEFT)
- QLoRA (4-bit quantization)
- Fast LLM training using Unsloth
- Supervised Fine-Tuning via TRL's `SFTTrainer`
- Beam search decoding
- BERTScore evaluation
- Deployment on Hugging Face Spaces

It is designed to demonstrate practical LLM training and production deployment skills.

---

# 🧠 Model Details

- **Base Model:** LLaMA-3.2-3B
- **Fine-Tuning Method:** QLoRA (4-bit)
- **Framework:** Unsloth + TRL SFTTrainer
- **Dataset:** CNN/DailyMail 3.0.0
- **Epochs:** 2
- **Inference Strategy:** Beam Search (num_beams=3)
- **Task:** Abstractive Text Summarization

---

# 🏋️ Training Setup

Training was performed using:

- unsloth.FastLanguageModel
- trl.SFTTrainer
- trl.SFTConfig
- PEFT (LoRA adapters)
- bitsandbytes (4-bit quantization)

### Why QLoRA?

- Reduces memory usage significantly
- Enables training large models on limited GPUs
- Maintains competitive performance
- Efficient fine-tuning without modifying base weights

### Training Characteristics

- 4-bit quantized base model
- LoRA adapters trained on attention layers
- Supervised Fine-Tuning (SFT)
- Loss masking applied only to response tokens
- Optimized for speed using Unsloth

---

# 📊 Evaluation

Evaluation was conducted on **200 samples** from the CNN/DailyMail test set using **BERTScore**.

### Results:

- **Precision:** 0.786  
- **Recall:** 0.816  
- **F1 Score:** 0.801  

### Why BERTScore?

- Embedding-based semantic similarity
- More robust than ROUGE for abstractive summarization
- Captures contextual meaning instead of surface overlap

---

# 🔍 Inference Strategy

Beam search was used during generation:

num_beams = 3

### Why Beam Search?

- Improves coherence
- More robust than ROUGE for abstractive summarization
- Captures contextual meaning instead of surface overlap
- More stable than greedy decoding, as it calculates probability of entire sequence than token
- Selects the highest probability summary

---

# 🌐 Deployment

The model is deployed using:

- Gradio
- Hugging Face Spaces
- Quantized base model + LoRA adapters

## 🔗 Live App:
**https://huggingface.co/spaces/osamakhaledML9/LLaMA-3.0-3B-eng_summarizer_app**
