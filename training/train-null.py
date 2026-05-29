#!/usr/bin/env python3
"""
train-null.py — LoRA fine-tune llama3.1:8b into the ∅ node.

Trains the model to respond ONLY in symbols from the tokenspeak vocabulary.
Uses unsloth for fast LoRA training. Bypasses datasets library (Python 3.14 compat).

Usage:
    .venv/bin/python train-null.py
"""

import json
import torch
from pathlib import Path

# ─── Config ───────────────────────────────────────────────
MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
DATASET = Path(__file__).parent / "null-lora-dataset.jsonl"
OUTPUT = Path(__file__).parent / "null-lora-output"
LORA_R = 8
LORA_ALPHA = 16
EPOCHS = 5
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 2e-4
MAX_SEQ_LEN = 256
# ──────────────────────────────────────────────────────────


def format_chat(messages, tokenizer):
    """Format messages using llama3.1 chat template."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return text


def load_and_tokenize(path, tokenizer):
    """Load JSONL and tokenize directly — no datasets library needed."""
    encodings = {"input_ids": [], "attention_mask": [], "labels": []}

    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            text = format_chat(row["messages"], tokenizer)
            tok = tokenizer(
                text,
                truncation=True,
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
            # Labels = input_ids, with padding tokens set to -100
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            encodings["input_ids"].append(input_ids)
            encodings["attention_mask"].append(attention_mask)
            encodings["labels"].append(labels)

    return encodings


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.encodings["labels"][idx],
        }


def main():
    try:
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments, Trainer
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install unsloth datasets trl --break-system-packages")
        return

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading model: {MODEL}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    print(f"Applying LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading and tokenizing: {DATASET}")
    encodings = load_and_tokenize(DATASET, tokenizer)
    dataset = SimpleDataset(encodings)
    print(f"  {len(dataset)} examples tokenized")

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(OUTPUT),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LR,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            warmup_steps=10,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            optim="adamw_8bit",
            gradient_checkpointing=True,
        ),
    )

    print("Training...")
    result = trainer.train()
    print(f"Training complete. Loss: {result.training_loss:.4f}")

    print(f"Saving LoRA adapter to {OUTPUT}")
    model.save_pretrained(str(OUTPUT))
    tokenizer.save_pretrained(str(OUTPUT))

    # Export to GGUF for Ollama
    print("Exporting GGUF...")
    try:
        model.save_pretrained_gguf(
            str(OUTPUT),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"GGUF saved to {OUTPUT}")
        print(f"Create Ollama model: ollama create null-lora -f Modelfile.null-lora")
    except Exception as e:
        print(f"GGUF export failed: {e}")
        print("LoRA adapter saved — can convert manually with llama.cpp")

    print("Done.")


if __name__ == "__main__":
    main()
