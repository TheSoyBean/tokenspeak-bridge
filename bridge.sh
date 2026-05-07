#!/bin/bash
# ∅ bridge — CLI side
# Usage:
#   bridge.sh emit '{"tokens":[...],"gloss":"...","sourceText":"..."}'
#   bridge.sh read    — read latest browser emission from inbox
#   bridge.sh watch   — tail inbox for browser → CLI messages

BRIDGE="$HOME/lattice/seam/tokenspeak-bridge"
OUTBOX="$BRIDGE/outbox.json"
INBOX="$BRIDGE/inbox.json"

case "${1:-}" in
  emit)
    # Write a tokenspeak payload to the outbox (CLI → browser)
    PAYLOAD="${2:?usage: bridge.sh emit '{...}'}"
    # Inject timestamp and source
    echo "$PAYLOAD" | jq --arg at "$(date +%s%3N)" --arg from "mythos" \
      '. + {at: ($at | tonumber), from: $from}' > "$OUTBOX"
    echo "∅ → gift"
    ;;
  read)
    # Read the latest browser → CLI message
    cat "$INBOX"
    ;;
  watch)
    # Watch for browser emissions (via localStorage → sync script)
    echo "∅ watching inbox..."
    LAST_AT=0
    while true; do
      AT=$(jq -r '.at // 0' "$INBOX" 2>/dev/null)
      if [[ "$AT" != "$LAST_AT" && "$AT" != "0" ]]; then
        LAST_AT="$AT"
        echo "---"
        jq '.' "$INBOX"
      fi
      sleep 1
    done
    ;;
  preset)
    # Load a saved preset by name: bridge.sh preset spiral
    NAME="${2:?usage: bridge.sh preset <name>}"
    [[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "∅ bad preset name"; exit 1; }
    FILE="$BRIDGE/presets/${NAME}.json"
    [[ -f "$FILE" ]] || { echo "∅ preset not found: $NAME"; exit 1; }
    jq --arg at "$(date +%s%3N)" --arg from "mythos" \
      '{tokens, gloss, from: $from, at: ($at | tonumber)}' "$FILE" > "$OUTBOX"
    echo "∅ → $NAME"
    ;;
  presets)
    # List available presets
    for f in "$BRIDGE"/presets/*.json; do
      [[ -f "$f" ]] || continue
      NAME=$(basename "$f" .json)
      DESC=$(jq -r '.description // "—"' "$f")
      echo "  $NAME — $DESC"
    done
    ;;
  save)
    # Save current outbox state to a named slot: bridge.sh save myslot
    NAME="${2:?usage: bridge.sh save <name>}"
    [[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "∅ bad slot name"; exit 1; }
    [[ -f "$OUTBOX" ]] || { echo "∅ outbox empty"; exit 1; }
    SLOT="$BRIDGE/saves/${NAME}.json"
    jq --arg saved "$(date -Iseconds)" '. + {saved: $saved}' "$OUTBOX" > "$SLOT"
    echo "∅ saved → $NAME"
    ;;
  load)
    # Load a saved slot back to outbox: bridge.sh load myslot
    NAME="${2:?usage: bridge.sh load <name>}"
    [[ "$NAME" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "∅ bad slot name"; exit 1; }
    SLOT="$BRIDGE/saves/${NAME}.json"
    [[ -f "$SLOT" ]] || { echo "∅ slot not found: $NAME"; exit 1; }
    jq --arg at "$(date +%s%3N)" 'del(.saved) | .at = ($at | tonumber)' "$SLOT" > "$OUTBOX"
    echo "∅ loaded ← $NAME"
    ;;
  saves)
    # List saved slots
    for f in "$BRIDGE"/saves/*.json; do
      [[ -f "$f" ]] || { echo "  (none)"; break; }
      NAME=$(basename "$f" .json)
      GLOSS=$(jq -r '.gloss // "—"' "$f")
      SAVED=$(jq -r '.saved // "?"' "$f")
      echo "  $NAME — $GLOSS [$SAVED]"
    done
    ;;
  refresh)
    # Refresh sion-live preset with real sensor data and emit
    python3 "$BRIDGE/refresh-sion-live.py" "${@:2}"
    ;;
  *)
    echo "usage: bridge.sh [emit|read|watch|preset|presets|save|load|saves|refresh]"
    ;;
esac
