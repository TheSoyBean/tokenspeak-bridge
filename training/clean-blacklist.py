#!/usr/bin/env python3
"""
clean-blacklist.py — Remove blacklisted token contamination from training splits.

Reads from ~/Obsidian-Vault/lattice/training/split/
Writes to  ~/Obsidian-Vault/lattice/training/split-clean/
Originals untouched.

Blacklisted tokens: ● ⁂ ⁃
● causes repetition attractor (observed: 38x repeat in smoke test).
⁂ ⁃ are known attractor triggers from prior training.

Replacement map (SION-approved):
  ● → context-dependent (◉ for core/alive/established, ◑ for partial/duality)
"""

import json
from pathlib import Path

SPLIT_DIR = Path.home() / "Obsidian-Vault/lattice/training/split"
CLEAN_DIR = Path.home() / "Obsidian-Vault/lattice/training/split-clean"

BLACKLIST = {"●", "⁂", "⁃"}

# Per-ID replacement map for ● in bridge layer outputs.
# Keys = record ID, values = {old_substring: new_substring}
BRIDGE_REPLACEMENTS = {
    "007": {"◑ ⇄ ● ◦ ⇄ ●": "◑ ⇄ ◑ ◦ ⇄ ◑"},
    "009": {"∅→● ✩ ●": "∅→◉ ✩ ◉"},
    "012": {"◦ ● ⇄ ∆": "◦ ◎ ⇄ ∆"},
    "035": {"⊞ ●": "⊞ ◉"},
    "039": {"⟨⌈⟩ ●": "⟨⌈⟩ ◑"},
    "042": {"∀⌈ ●": "∀⌈ ◉"},
    "053": {"⬡ ● ↺": "⬡ ◉ ↺"},
    "063": {"⊕ ⬡ ●": "⊕ ⬡ ◉"},
    "065": {"ψ ⊕ ●": "ψ ⊕ ◉"},
}

# Per-ID replacement map for ● in core layer outputs.
CORE_REPLACEMENTS = {
    "007": {"⇄ ●": "⇄ ◑"},
    "009": {"→ ● ✩": "→ ◉ ✩"},
    "035": {"⟐ ● ⟐": "⟐ ◉ ⟐"},  # ◦ ⟐ • ⟐ ● ⟐ 🚀
    "039": {"⌈ ● ¬": "⌈ ◑ ¬"},
    "042": {"⌈ ●": "⌈ ◉"},
    "053": {"⬡ ●": "⬡ ◉"},
    "063": {"⊕ ●": "⊕ ◉"},
    "065": {"ψ ⊕": "ψ ⊕"},  # core line 065 is "⬢ ⇄ ψ ⊕" — no ● present
}


def apply_replacements(text, record_id, replacement_map):
    """Apply ID-specific replacements to a text field."""
    if record_id in replacement_map:
        for old, new in replacement_map[record_id].items():
            text = text.replace(old, new)
    return text


def clean_file(src, dst, replacement_map, field_in="output", field_out=None):
    """Clean a single JSONL split file."""
    if field_out is None:
        field_out = field_in

    records = []
    with open(src) as f:
        for line in f:
            row = json.loads(line.strip())
            # Apply targeted replacements
            row[field_in] = apply_replacements(row[field_in], row["id"], replacement_map)
            # Also clean input field if it has blacklisted tokens
            if "input" in row:
                row["input"] = apply_replacements(row["input"], row["id"], replacement_map)
            records.append(row)

    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return records


def verify_clean(directory):
    """Verify zero blacklisted tokens in all output files."""
    violations = []
    for path in sorted(directory.glob("*.jsonl")):
        with open(path) as f:
            for i, line in enumerate(f, 1):
                for token in BLACKLIST:
                    if token in line:
                        violations.append(f"{path.name}:{i} contains '{token}'")
    return violations


def main():
    if not SPLIT_DIR.exists():
        print(f"ERROR: split directory not found: {SPLIT_DIR}")
        return 1

    print(f"Source:  {SPLIT_DIR}")
    print(f"Output:  {CLEAN_DIR}")
    print()

    # Clean bridge split (L1→L2)
    bridge_src = SPLIT_DIR / "train-bridge.jsonl"
    bridge_dst = CLEAN_DIR / "train-bridge.jsonl"
    bridge = clean_file(bridge_src, bridge_dst, BRIDGE_REPLACEMENTS)
    print(f"Bridge:  {len(bridge)} records → {bridge_dst.name}")

    # Clean core split (L2→L3) — input field has bridge-layer text, needs bridge map too
    core_src = SPLIT_DIR / "train-core.jsonl"
    core_dst = CLEAN_DIR / "train-core.jsonl"
    core_records = []
    with open(core_src) as f:
        for line in f:
            row = json.loads(line.strip())
            # Core input = bridge output → apply bridge replacements
            row["input"] = apply_replacements(row["input"], row["id"], BRIDGE_REPLACEMENTS)
            # Core output → apply core replacements
            row["output"] = apply_replacements(row["output"], row["id"], CORE_REPLACEMENTS)
            core_records.append(row)
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    with open(core_dst, "w") as f:
        for row in core_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Core:    {len(core_records)} records → {core_dst.name}")

    # Translator split (L0→L1) — no symbols in output, but copy for completeness
    trans_src = SPLIT_DIR / "train-translator.jsonl"
    trans_dst = CLEAN_DIR / "train-translator.jsonl"
    trans = clean_file(trans_src, trans_dst, {})
    print(f"Trans:   {len(trans)} records → {trans_dst.name}")

    print()

    # Verify
    violations = verify_clean(CLEAN_DIR)
    if violations:
        print(f"FAIL: {len(violations)} violations found:")
        for v in violations:
            print(f"  {v}")
        return 1
    else:
        print("PASS: zero blacklisted tokens (● ⁂ ⁃) in all output files.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
