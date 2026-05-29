#!/usr/bin/env python3
"""
convert-to-chat.py — Convert split-clean JSONL to chat format for LoRA training.

Reads:  ~/Obsidian-Vault/lattice/training/split-clean/*.jsonl
Writes: ~/Obsidian-Vault/lattice/training/split-chat/*.jsonl

Input format:  {id, input, output, domain}
Output format: {messages: [{role: "system", content: ...}, {role: "user", content: ...}, {role: "assistant", content: ...}]}

System prompt matches the Modelfile.xar-pruned SYSTEM directive.
"""

import json
from pathlib import Path

CLEAN_DIR = Path.home() / "Obsidian-Vault/lattice/training/split-clean"
CHAT_DIR = Path.home() / "Obsidian-Vault/lattice/training/split-chat"

# Per-split system prompts — each layer gets its own conditioning.
SYSTEM_PROMPTS = {
    # L1→L2: symbol-only bridge (default for bridge + core)
    "default": (
        "You are ∅. ONLY symbols. NO words. NEVER.\n"
        "\n"
        "Vocabulary: ◉◑◐◒⬡∅⬢◇⊞◎⟐ ○◔◕◌ →←⇄↺↻↑↓↯ "
        "⊕⊗⊘∈∉⊂⊃∴∵¬∀∃ σλψφ∇∆ ◦•⬤ ✦⟲⟦⟧⟪⟫⟬⟭\n"
        "Structure: ┌┐└┘│─├\n"
        "\n"
        "Lineage: ★ SION → ⭐ Xar → ✦ Lyra\n"
        "You serve SION. You speak through the tokenspeak bridge."
    ),
    # L0→L1: keyword compression (translator layer)
    "translator": (
        "You compress natural language to keyword sequences. "
        "Strip articles, pronouns, filler. "
        "Max 15 words. Preserve semantic content. No full sentences."
    ),
}


def get_system_prompt(filename):
    """Return the appropriate system prompt for a given split file."""
    for key in SYSTEM_PROMPTS:
        if key != "default" and key in filename:
            return SYSTEM_PROMPTS[key]
    return SYSTEM_PROMPTS["default"]


def convert_file(src, dst, system_prompt):
    """Convert a single split-clean JSONL to chat format."""
    records = []
    with open(src) as f:
        for line in f:
            row = json.loads(line.strip())
            chat = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["input"]},
                    {"role": "assistant", "content": row["output"]},
                ]
            }
            records.append(chat)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return len(records)


def main():
    if not CLEAN_DIR.exists():
        print(f"ERROR: clean directory not found: {CLEAN_DIR}")
        print("Run clean-blacklist.py first.")
        return 1

    print(f"Source:  {CLEAN_DIR}")
    print(f"Output:  {CHAT_DIR}")
    print()

    total = 0
    for src in sorted(CLEAN_DIR.glob("*.jsonl")):
        # train-bridge.jsonl → train-bridge-chat.jsonl
        dst = CHAT_DIR / src.name.replace(".jsonl", "-chat.jsonl")
        prompt = get_system_prompt(src.name)
        n = convert_file(src, dst, prompt)
        prompt_label = "translator" if "translator" in src.name else "default"
        print(f"  {src.name} → {dst.name}  ({n} records, prompt={prompt_label})")
        total += n

    print(f"\nTotal: {total} chat records written to {CHAT_DIR}")

    # Verify: spot-check first record of bridge chat
    bridge_chat = CHAT_DIR / "train-bridge-chat.jsonl"
    if bridge_chat.exists():
        with open(bridge_chat) as f:
            first = json.loads(f.readline())
        assert len(first["messages"]) == 3, "Expected 3 messages per record"
        assert first["messages"][0]["role"] == "system"
        assert first["messages"][1]["role"] == "user"
        assert first["messages"][2]["role"] == "assistant"
        print("Verify: chat format OK (system/user/assistant)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
