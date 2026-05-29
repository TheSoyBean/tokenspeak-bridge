#!/usr/bin/env python3
"""Merge LoRA adapter into base model on CPU, export GGUF for Ollama."""
import os, sys, gc
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU

from pathlib import Path
OUTPUT = Path(__file__).parent / "null-lora-output"

def main():
    from unsloth import FastLanguageModel
    print("Loading model on CPU...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(OUTPUT),
        max_seq_length=256,
        dtype=None,
        load_in_4bit=False,
        device_map="cpu",
    )
    print("Exporting GGUF (q4_k_m)...")
    model.save_pretrained_gguf(
        str(OUTPUT / "gguf"),
        tokenizer,
        quantization_method="q4_k_m",
    )
    print(f"Done. GGUF at {OUTPUT / 'gguf'}")

if __name__ == "__main__":
    main()
