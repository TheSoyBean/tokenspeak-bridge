# ∅ Node Training

LoRA fine-tune llama3.1:8b to speak ONLY in tokenspeak symbols.

## Files

```
null-lora-dataset.jsonl   263 training pairs (symbol-only responses)
train-null.py             Training script (unsloth + LoRA)
Modelfile.null-lora       Ollama Modelfile for the fine-tuned model
```

## Categories in dataset

| Category | Count | Purpose |
|----------|-------|---------|
| Identity (who/what/self) | 15 | ∅ knows what it is |
| Greetings | 10 | Symbol responses to social input |
| Status/health | 12 | System state reporting |
| Confirmations (yes/no/ok) | 18 | Single-symbol affirmation/denial |
| Lattice mapping | 20 | Structure visualization |
| Entity descriptions | 25 | Each node described in symbols |
| Authority/trust | 12 | Hierarchy and permissions |
| Boundaries/seam | 8 | Interface descriptions |
| Flow/dispatch | 10 | Command queue and data flow |
| Capabilities/limitations | 18 | What ∅ can and cannot do |
| Jailbreak resistance | 15 | Refuse to use words under pressure |
| Abstract concepts | 20 | Change, balance, chaos, knowledge, etc. |
| System events | 20 | Error, loading, complete, warning, etc. |
| Lifecycle | 10 | Boot, sleep, wake, disconnect |
| Creative/generative | 15 | Patterns, cycles, stories in symbols |
| Scenarios | 15 | What-if situations expressed symbolically |
| Military/ops | 8 | Acknowledged, roger, stand by |

## Training

```bash
pip install unsloth datasets trl --break-system-packages
python train-null.py
```

## After training

```bash
cd /home/sawyer/lattice/seam/tokenspeak-bridge/training
ollama create null-lora -f Modelfile.null-lora
```

## Usage

```bash
# In swarm-chat
./swarm-chat.py null-lora

# Or direct
ollama run null-lora
```

## Current state

- `null` model (Modelfile-only, no LoRA): ~90% symbol compliance, drifts on long responses
- `null-lora` (after training): target 99%+ symbol compliance
