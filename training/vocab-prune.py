#!/usr/bin/env python3
"""
vocab-prune.py — Vocabulary pruning pipeline for compressed tokenspeak LLM.

Takes a pretrained model (llama3.1:8b) and surgically removes all tokens
not in the tokenspeak symbol set from the embedding and lm_head layers.
Then fine-tunes via LoRA so the model learns to reason through the bottleneck.

This is NOT output conditioning (like null-lora). This physically removes
vocabulary entries — the model literally cannot produce tokens outside the set.

Pipeline:
  1. Load base model + tokenizer
  2. Build allowed token set from null-vocab.json + structural chars
  3. Prune embedding (model.embed_tokens) and lm_head to allowed set only
  4. Fine-tune with LoRA on existing training data
  5. Export to GGUF for ollama

Usage:
    .venv/bin/python vocab-prune.py [--dry-run] [--analyze-only]
"""

import argparse
import json
import os
import sys
import gc
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn

# ─── Config ───────────────────────────────────────────────
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
VOCAB_FILE = Path(__file__).parent.parent / "null-vocab.json"
DATASET = Path(__file__).parent / "null-lora-dataset.jsonl"
OUTPUT = Path(__file__).parent / "pruned-output"
MAX_SEQ_LEN = 512  # longer than null-lora — pruned model needs more room
LORA_R = 16         # higher rank for harder task
LORA_ALPHA = 32
EPOCHS = 10
BATCH_SIZE = 1
GRAD_ACCUM = 8
LR = 1e-4           # lower LR for stability after pruning
# ──────────────────────────────────────────────────────────


def load_tokenspeak_symbols(vocab_path):
    """Load all symbols from null-vocab.json into a flat set."""
    with open(vocab_path) as f:
        vocab = json.load(f)

    symbols = set()
    for category, entries in vocab.items():
        if category.startswith("_"):
            continue
        if isinstance(entries, dict):
            for sym, desc in entries.items():
                # Each key might be a single symbol or a compound like "┌┐└┘│─"
                for ch in sym:
                    symbols.add(ch)

    return symbols


def build_allowed_token_ids(tokenizer, symbols):
    """Map tokenspeak symbols to token IDs in the base tokenizer.

    Strategy:
    - For each symbol, find ALL token IDs that encode it (alone or as part)
    - Also include structural tokens: BOS, EOS, PAD, newline, space
    - Also include digits 0-9 for grid coordinates
    - Also include basic punctuation for JSON output: {}[],:""
    """
    allowed_ids = set()

    # Always keep special tokens
    special = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    ]
    for sid in special:
        if sid is not None:
            allowed_ids.add(sid)

    # Keep structural characters needed for communication
    structural_chars = set()
    # Digits for coordinates
    structural_chars.update("0123456789")
    # JSON structure
    structural_chars.update('{}[],:." ')
    # Newlines and whitespace
    structural_chars.update("\n\t")
    # Arrows and operators that might be multi-byte in tokenizer
    structural_chars.update("+-*/=<>()_|&!?;'\\/@#$%^~`")

    all_chars = symbols | structural_chars

    # Method 1: Direct single-character encoding
    for ch in all_chars:
        encoded = tokenizer.encode(ch, add_special_tokens=False)
        allowed_ids.update(encoded)

    # Method 2: Check every token in vocab for symbol content
    vocab_size = len(tokenizer)
    for tid in range(vocab_size):
        try:
            decoded = tokenizer.decode([tid])
            # Keep token if it decodes to ONLY allowed characters
            if decoded and all(c in all_chars for c in decoded):
                allowed_ids.add(tid)
        except Exception:
            continue

    # Method 3: Encode full symbol sequences from training data
    if DATASET.exists():
        with open(DATASET) as f:
            for line in f:
                row = json.loads(line.strip())
                for msg in row["messages"]:
                    if msg["role"] == "assistant":
                        encoded = tokenizer.encode(
                            msg["content"], add_special_tokens=False
                        )
                        allowed_ids.update(encoded)

    return sorted(allowed_ids)


def analyze_pruning(tokenizer, allowed_ids):
    """Print analysis of what we're keeping vs removing."""
    total = len(tokenizer)
    kept = len(allowed_ids)
    removed = total - kept
    ratio = kept / total * 100

    print(f"\n{'='*60}")
    print(f"VOCABULARY PRUNING ANALYSIS")
    print(f"{'='*60}")
    print(f"Base vocabulary:    {total:>8,} tokens")
    print(f"Kept after pruning: {kept:>8,} tokens ({ratio:.1f}%)")
    print(f"Removed:            {removed:>8,} tokens ({100-ratio:.1f}%)")
    print(f"Compression ratio:  {total/kept:>8.1f}x")
    print()

    # Show what we're keeping, grouped
    categories = defaultdict(list)
    for tid in allowed_ids[:500]:  # sample
        decoded = tokenizer.decode([tid])
        if decoded.strip() == "":
            categories["whitespace"].append(repr(decoded))
        elif decoded.isdigit():
            categories["digits"].append(decoded)
        elif decoded.isascii() and decoded.isprintable():
            categories["ascii"].append(decoded)
        else:
            categories["symbols"].append(decoded)

    for cat, tokens in sorted(categories.items()):
        unique = sorted(set(tokens))
        print(f"  {cat}: {len(unique)} unique — {' '.join(unique[:30])}")

    # Embedding size savings
    # Original: vocab_size × hidden_dim × 2 (embed + lm_head) × dtype_bytes
    hidden = 4096  # llama3.1:8b
    orig_params = total * hidden * 2
    new_params = kept * hidden * 2
    saved_mb = (orig_params - new_params) * 2 / 1024 / 1024  # fp16

    print(f"\n  Embedding parameter reduction:")
    print(f"    Original: {orig_params:>12,} params ({orig_params * 2 / 1024/1024:.0f} MB fp16)")
    print(f"    Pruned:   {new_params:>12,} params ({new_params * 2 / 1024/1024:.0f} MB fp16)")
    print(f"    Saved:    {saved_mb:.0f} MB")
    print(f"{'='*60}\n")

    return kept, removed


def prune_embeddings(model, tokenizer, allowed_ids):
    """Surgically prune embedding and lm_head to only allowed tokens.

    Creates a new, smaller embedding table and output projection.
    Maps old token IDs → new contiguous IDs.
    """
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(allowed_ids)}
    new_vocab_size = len(allowed_ids)

    # Get original layers
    old_embed = model.model.embed_tokens
    old_lm_head = model.lm_head
    hidden_dim = old_embed.embedding_dim

    print(f"Pruning embeddings: {old_embed.num_embeddings} → {new_vocab_size}")

    # Create new embedding layer
    new_embed = nn.Embedding(new_vocab_size, hidden_dim, dtype=old_embed.weight.dtype)
    with torch.no_grad():
        for new_id, old_id in enumerate(allowed_ids):
            new_embed.weight[new_id] = old_embed.weight[old_id]

    # Create new lm_head
    new_lm_head = nn.Linear(hidden_dim, new_vocab_size, bias=False, dtype=old_lm_head.weight.dtype)
    with torch.no_grad():
        for new_id, old_id in enumerate(allowed_ids):
            new_lm_head.weight[new_id] = old_lm_head.weight[old_id]

    # Replace in model
    model.model.embed_tokens = new_embed
    model.lm_head = new_lm_head
    model.config.vocab_size = new_vocab_size

    # Remap tokenizer
    # We need to create a mapping so the tokenizer outputs new IDs
    print(f"  Embedding: {old_embed.num_embeddings}×{hidden_dim} → {new_vocab_size}×{hidden_dim}")
    print(f"  lm_head:   {old_lm_head.out_features} → {new_vocab_size}")

    return old_to_new


def create_id_remapper(old_to_new, tokenizer):
    """Create a wrapper that remaps tokenizer output to new ID space."""

    class RemappedTokenizer:
        """Wraps a tokenizer to remap IDs to pruned vocabulary."""

        def __init__(self, base_tokenizer, mapping):
            self._base = base_tokenizer
            self._old_to_new = mapping
            self._new_to_old = {v: k for k, v in mapping.items()}
            # Copy attributes
            self.pad_token = base_tokenizer.pad_token
            self.eos_token = base_tokenizer.eos_token
            self.bos_token = base_tokenizer.bos_token
            self.pad_token_id = mapping.get(base_tokenizer.pad_token_id, 0)
            self.eos_token_id = mapping.get(base_tokenizer.eos_token_id, 0)
            self.bos_token_id = mapping.get(base_tokenizer.bos_token_id, 0)

        def __len__(self):
            return len(self._old_to_new)

        def __call__(self, *args, **kwargs):
            result = self._base(*args, **kwargs)
            # Remap input_ids
            if "input_ids" in result:
                remapped = []
                for old_id in result["input_ids"].squeeze().tolist():
                    new_id = self._old_to_new.get(old_id, self.pad_token_id)
                    remapped.append(new_id)
                result["input_ids"] = torch.tensor([remapped])
            return result

        def encode(self, text, **kwargs):
            old_ids = self._base.encode(text, **kwargs)
            return [self._old_to_new.get(oid, self.pad_token_id) for oid in old_ids]

        def decode(self, ids, **kwargs):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            old_ids = [self._new_to_old.get(nid, 0) for nid in ids]
            return self._base.decode(old_ids, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            return self._base.apply_chat_template(*args, **kwargs)

        def save_pretrained(self, path):
            self._base.save_pretrained(path)
            # Also save the mapping
            mapping_path = Path(path) / "token_mapping.json"
            mapping_path.write_text(json.dumps({
                "old_to_new": {str(k): v for k, v in self._old_to_new.items()},
                "new_to_old": {str(k): v for k, v in self._new_to_old.items()},
                "pruned_vocab_size": len(self._old_to_new),
            }, indent=2))

    return RemappedTokenizer(tokenizer, old_to_new)


def load_and_tokenize_remapped(path, tokenizer, max_seq_len):
    """Load JSONL and tokenize with remapped tokenizer."""
    encodings = {"input_ids": [], "attention_mask": [], "labels": []}

    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False
            )
            tok = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_len,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = tok["input_ids"].squeeze(0)
            attention_mask = tok["attention_mask"].squeeze(0)
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
    parser = argparse.ArgumentParser(description="Vocabulary pruning for tokenspeak LLM")
    parser.add_argument("--dry-run", action="store_true", help="Analyze only, don't modify model")
    parser.add_argument("--analyze-only", action="store_true", help="Just show pruning stats")
    parser.add_argument("--skip-train", action="store_true", help="Prune but don't fine-tune")
    args = parser.parse_args()

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Step 1: Load symbols
    print("Loading tokenspeak vocabulary...")
    symbols = load_tokenspeak_symbols(VOCAB_FILE)
    print(f"  {len(symbols)} unique symbols from null-vocab.json")

    # Step 2: Load model + tokenizer
    print(f"\nLoading base model: {BASE_MODEL}")
    try:
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments, Trainer
    except ImportError as e:
        print(f"Missing: {e}")
        print("Install: pip install unsloth datasets trl")
        return

    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Step 3: Build allowed token set
    print("\nBuilding allowed token set...")
    allowed_ids = build_allowed_token_ids(tokenizer, symbols)

    # Step 4: Analyze
    kept, removed = analyze_pruning(tokenizer, allowed_ids)

    if args.dry_run or args.analyze_only:
        print("Dry run complete. No changes made.")
        return

    # Step 5: Prune embeddings
    print("Pruning model embeddings...")
    old_to_new = prune_embeddings(model, tokenizer, allowed_ids)
    remapped_tokenizer = create_id_remapper(old_to_new, tokenizer)

    gc.collect()
    torch.cuda.empty_cache()

    if args.skip_train:
        print("Skipping training. Saving pruned model...")
        OUTPUT.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(OUTPUT))
        remapped_tokenizer.save_pretrained(str(OUTPUT))
        print(f"Saved to {OUTPUT}")
        return

    # Step 6: Apply LoRA for fine-tuning
    print(f"\nApplying LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
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

    # Step 7: Load and tokenize training data with remapped IDs
    print(f"\nLoading training data: {DATASET}")
    encodings = load_and_tokenize_remapped(DATASET, remapped_tokenizer, MAX_SEQ_LEN)
    dataset = SimpleDataset(encodings)
    print(f"  {len(dataset)} examples tokenized with pruned vocabulary")

    # Step 8: Train
    OUTPUT.mkdir(parents=True, exist_ok=True)
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
            logging_steps=5,
            save_strategy="epoch",
            warmup_steps=20,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            report_to="none",
            optim="adamw_8bit",
            gradient_checkpointing=True,
        ),
    )

    print("\nTraining on pruned vocabulary...")
    result = trainer.train()
    print(f"Training complete. Loss: {result.training_loss:.4f}")

    # Step 9: Save
    print(f"\nSaving to {OUTPUT}")
    model.save_pretrained(str(OUTPUT))
    remapped_tokenizer.save_pretrained(str(OUTPUT))

    # Save metadata
    meta = {
        "base_model": BASE_MODEL,
        "original_vocab_size": len(tokenizer),
        "pruned_vocab_size": len(allowed_ids),
        "compression_ratio": len(tokenizer) / len(allowed_ids),
        "symbols_from": str(VOCAB_FILE),
        "training_data": str(DATASET),
        "training_examples": len(dataset),
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "epochs": EPOCHS,
        "final_loss": result.training_loss,
    }
    (OUTPUT / "pruning_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Metadata saved to {OUTPUT / 'pruning_meta.json'}")

    print("\nDone. Next steps:")
    print(f"  1. Test: .venv/bin/python test-pruned.py")
    print(f"  2. Export GGUF: .venv/bin/python export-pruned-gguf.py")
    print(f"  3. Deploy: ollama create xar-pruned -f Modelfile.pruned")


if __name__ == "__main__":
    main()
