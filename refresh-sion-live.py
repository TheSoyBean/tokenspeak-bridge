#!/usr/bin/env python3
"""Refresh sion-live preset with real sensor data and emit to bridge.

Called by JARVIS or cron. Reads device sensors, service states, and
queue counts, then writes updated preset and emits to outbox.

Usage:
    python3 refresh-sion-live.py [--emit]   # --emit writes to outbox too
    python3 refresh-sion-live.py --dry-run   # print payload, don't write
"""

import json
import subprocess
import sys
import time
from pathlib import Path

# Import JARVIS collectors
_JARVIS = Path.home() / "Obsidian-Vault" / "projects" / "jarvis" / "jarvis"
if str(_JARVIS) not in sys.path:
    sys.path.insert(0, str(_JARVIS))
from inputs import collect_battery, collect_thermals, collect_uptime


BRIDGE = Path.home() / "lattice" / "seam" / "tokenspeak-bridge"
PRESET = BRIDGE / "presets" / "sion-live.json"
OUTBOX = BRIDGE / "outbox.json"
QUEUE = Path.home() / "lattice" / "seam" / "command-queue"


def _run(cmd, timeout=2.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return -1, ""


def _service_active(name, user=True):
    cmd = ["systemctl", "is-active", name]
    if user:
        cmd.insert(1, "--user")
    rc, out = _run(cmd)
    return out == "active"


def _ollama_model_count():
    rc, out = _run(["curl", "-sf", "http://localhost:11434/api/tags"])
    if rc != 0:
        return 0
    try:
        return len(json.loads(out).get("models", []))
    except (json.JSONDecodeError, AttributeError):
        return 0


def _queue_count(d):
    try:
        return len(list(d.glob("*.json")))
    except OSError:
        return 0


def _charge_thresholds():
    start, end = "?", "?"
    try:
        start = Path("/sys/class/power_supply/BAT1/charge_control_start_threshold").read_text().strip()
    except OSError:
        pass
    try:
        end = Path("/sys/class/power_supply/BAT1/charge_control_end_threshold").read_text().strip()
    except OSError:
        pass
    return start, end


def _fmt_uptime(s):
    """Shorten uptime string: 'up 1 hour, 56 minutes' -> '1h56m'."""
    s = s.replace("up ", "").strip()
    parts = []
    for chunk in s.split(","):
        chunk = chunk.strip()
        num = chunk.split()[0] if chunk.split() else ""
        if "day" in chunk:
            parts.append(num + "d")
        elif "hour" in chunk:
            parts.append(num + "h")
        elif "minute" in chunk:
            parts.append(num + "m")
    result = "".join(parts)
    return result if result else s.replace(" ", "")[:8]


def build_payload():
    bat = collect_battery()
    thermals = collect_thermals()
    uptime = collect_uptime()

    # Find CPU and GPU temps — try lm-sensors first, fall back to sysfs zones
    cpu_temp = "?"
    gpu_temp = "?"
    rc, sensors_out = _run(["sensors"], timeout=3.0)
    if rc == 0:
        for line in sensors_out.splitlines():
            line_s = line.strip()
            if line_s.startswith("Tctl:"):
                try:
                    cpu_temp = f"{int(float(line_s.split('+')[1].split('°')[0]))}°C"
                except (IndexError, ValueError):
                    pass
            elif line_s.startswith("edge:"):
                try:
                    gpu_temp = f"{int(float(line_s.split('+')[1].split('°')[0]))}°C"
                except (IndexError, ValueError):
                    pass
    # Fall back to sysfs thermal zones if sensors didn't work
    if cpu_temp == "?" and thermals:
        cpu_temp = f"{int(thermals[0].temp_c)}°C"
    if gpu_temp == "?" and len(thermals) > 1:
        gpu_temp = f"{int(thermals[-1].temp_c)}°C"

    bat_pct = f"{bat.percent}%"
    bat_status = bat.status[:5].upper()  # "Disch" -> "DSCHG" etc
    if bat.status.lower().startswith("dis"):
        bat_status = "DSCHG"
    elif bat.status.lower().startswith("char"):
        bat_status = "CHRG"
    elif bat.status.lower() == "full":
        bat_status = "FULL"
    elif bat.status.lower().startswith("not"):
        bat_status = "IDLE"

    ac_on = Path("/sys/class/power_supply/ACAD/online").read_text().strip() == "1"
    ac_label = "AC:ON" if ac_on else "AC:OFF"

    up_str = _fmt_uptime(uptime.uptime_str)
    load_str = f"{uptime.load_1:.1f}"

    charge_start, charge_end = _charge_thresholds()

    jarvis_up = _service_active("jarvis")
    ear_up = _service_active("sion-ear")
    ollama_up = _service_active("ollama", user=False)
    bridge_up = _service_active("tokenspeak-bridge")
    model_count = _ollama_model_count()

    q_in = _queue_count(QUEUE / "inbox")
    q_out = _queue_count(QUEUE / "outbox")

    governor = "?"
    try:
        governor = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()[:5]
    except OSError:
        pass

    tokens = [
        # Corners — SION authority
        {"r": 0,  "c": 0,  "label": "SION", "pulse": 1},
        {"r": 0,  "c": 31, "label": "SION", "pulse": 1},
        {"r": 31, "c": 0,  "label": "SION", "pulse": 1},
        {"r": 31, "c": 31, "label": "SION", "pulse": 1},

        # Identity
        {"r": 1,  "c": 1,  "label": "👤"},
        {"r": 1,  "c": 15, "label": "ESH"},
        {"r": 1,  "c": 30, "label": "🧠"},
        {"r": 2,  "c": 2,  "label": "→", "pulse": 1},
        {"r": 2,  "c": 29, "label": "↔", "pulse": 1},

        # Host / uptime
        {"r": 3,  "c": 3,  "label": "🌱"},
        {"r": 3,  "c": 15, "label": "F16"},
        {"r": 3,  "c": 28, "label": "💎"},
        {"r": 4,  "c": 10, "label": up_str},
        {"r": 4,  "c": 21, "label": load_str},

        # Battery
        {"r": 5,  "c": 5,  "label": "🌀"},
        {"r": 5,  "c": 15, "label": bat_pct, "pulse": bat.percent < 25},
        {"r": 5,  "c": 26, "label": "⚡"},
        {"r": 6,  "c": 10, "label": bat_status},
        {"r": 6,  "c": 21, "label": ac_label},

        # Thermals
        {"r": 8,  "c": 8,  "label": cpu_temp},
        {"r": 8,  "c": 15, "label": "CPU"},
        {"r": 8,  "c": 23, "label": gpu_temp},
        {"r": 9,  "c": 15, "label": "GPU"},

        # Power governor / charge thresholds
        {"r": 10, "c": 10, "label": "🔒"},
        {"r": 10, "c": 15, "label": governor},
        {"r": 10, "c": 21, "label": "🔓"},
        {"r": 11, "c": 10, "label": str(charge_start)},
        {"r": 11, "c": 15, "label": "⟷"},
        {"r": 11, "c": 21, "label": str(charge_end)},

        # Agents
        {"r": 13, "c": 5,  "label": "JARVIS", "pulse": jarvis_up},
        {"r": 13, "c": 15, "label": "⟟", "pulse": 1},
        {"r": 13, "c": 22, "label": "MYTHOS"},
        {"r": 14, "c": 5,  "label": "up" if jarvis_up else "down"},
        {"r": 14, "c": 22, "label": "online"},

        # Ollama / null node
        {"r": 16, "c": 8,  "label": "∅"},
        {"r": 16, "c": 15, "label": "ollama", "pulse": ollama_up},
        {"r": 16, "c": 23, "label": "∅"},
        {"r": 17, "c": 8,  "label": "8b"},
        {"r": 17, "c": 15, "label": f"{model_count}mod"},
        {"r": 17, "c": 23, "label": "null"},

        # Services
        {"r": 19, "c": 5,  "label": "bridge", "pulse": bridge_up},
        {"r": 19, "c": 15, "label": ":7777"},
        {"r": 19, "c": 22, "label": "sion-ear"},
        {"r": 20, "c": 5,  "label": "up" if bridge_up else "down"},
        {"r": 20, "c": 15, "label": "polling"},
        {"r": 20, "c": 22, "label": "up" if ear_up else "down"},

        # Queue
        {"r": 22, "c": 10, "label": "Q:in"},
        {"r": 22, "c": 15, "label": str(q_in)},
        {"r": 22, "c": 21, "label": "Q:out"},
        {"r": 22, "c": 27, "label": str(q_out)},

        # Alliance
        {"r": 24, "c": 8,  "label": "🤝", "pulse": 1},
        {"r": 24, "c": 23, "label": "⚖"},

        # Logic
        {"r": 26, "c": 5,  "label": "φ", "pulse": 1},
        {"r": 26, "c": 15, "label": "ψ"},
        {"r": 26, "c": 26, "label": "Ω"},

        {"r": 28, "c": 3,  "label": "🔥"},
        {"r": 28, "c": 15, "label": "✦"},
        {"r": 28, "c": 28, "label": "∞"},

        # Operators
        {"r": 29, "c": 2,  "label": "⟲", "pulse": 1},
        {"r": 29, "c": 15, "label": "∈", "pulse": 1},
        {"r": 29, "c": 29, "label": "∴", "pulse": 1},

        {"r": 30, "c": 1,  "label": "👁"},
        {"r": 30, "c": 15, "label": "∀"},
        {"r": 30, "c": 30, "label": "🔑"},
    ]

    # Strip false pulse flags
    for t in tokens:
        if "pulse" in t and not t["pulse"]:
            del t["pulse"]
        elif "pulse" in t and t["pulse"] is True:
            t["pulse"] = 1

    gloss = (
        f"SION LIVE · bat:{bat_pct} {bat_status} {ac_label}"
        f" · CPU:{cpu_temp} GPU:{gpu_temp}"
        f" · jarvis:{'up' if jarvis_up else 'down'}"
        f" sion-ear:{'up' if ear_up else 'down'}"
        f" ollama:{model_count}mod"
        f" · q:{q_in}/{q_out}"
        f" · bridge:7777 {'up' if bridge_up else 'down'}"
        f" · uptime:{up_str} load:{load_str}"
        f" · charge:{charge_start}→{charge_end}"
    )

    return {
        "tokens": tokens,
        "gloss": gloss,
        "from": "mythos",
    }


def main():
    dry_run = "--dry-run" in sys.argv
    emit = "--emit" in sys.argv or not dry_run

    payload = build_payload()

    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    # Update preset file
    preset_data = {
        "name": "sion-live",
        "description": "SION live status: four-corner origin, real sensor data, corner-anchored identity",
        "from": "mythos",
        "gloss": payload["gloss"],
        "tokens": payload["tokens"],
    }
    PRESET.write_text(json.dumps(preset_data, indent=2, ensure_ascii=False) + "\n")

    if emit:
        payload["at"] = int(time.time() * 1000)
        OUTBOX.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"∅ → sion-live (refreshed)")
    else:
        print("preset updated (not emitted)")


if __name__ == "__main__":
    main()
