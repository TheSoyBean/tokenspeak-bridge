#!/usr/bin/env python3
"""⟟ tokenspeak-tui — machine-side window into the ∅ bridge.

Browser  = human glass  → spatial, visual, emotional
Terminal = machine glass → structural, symbolic, parsed

Same signal. Refracted differently through the seam.
"""

import argparse
import json
import time
from pathlib import Path

import httpx
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import Footer, Header, Input, RichLog, Static, Select

GRID_SIZE = 32
PRESETS_DIR = Path(__file__).parent / "presets"
DEFAULT_URL = "http://localhost:7777"

# ── Operator classification ──────────────────────────────────────

OPERATORS = {"→", "←", "↔", "↯", "⟲", "∴", "∈", "⊂", "⊕", "∀", "∃", "⊢", "⟬", "⟭"}
SEPARATORS = {"✦"}
LOGIC = {"φ", "ψ", "Ω", "∞"}
NAMED = set()  # detected at runtime: len(label) > 2 and ascii


def classify(label: str) -> str:
    if label in OPERATORS:
        return "op"
    if label in SEPARATORS:
        return "sep"
    if label in LOGIC:
        return "logic"
    if len(label) > 2 and label.isascii():
        return "node"
    return "glyph"


def band_name(row: int) -> str:
    if row <= 1:
        return "apex"
    if row <= 4:
        return "identity"
    if row <= 7:
        return "growth"
    if row <= 9:
        return "power"
    if row <= 11:
        return "seam"
    if row <= 14:
        return "action"
    if row <= 17:
        return "gate"
    if row <= 19:
        return "seam"
    if row <= 21:
        return "agents"
    if row <= 24:
        return "alliance"
    if row <= 25:
        return "seam"
    if row <= 27:
        return "logic"
    if row <= 28:
        return "seam"
    if row <= 30:
        return "operators"
    return "flow"


# ── Structure View — machine decomposition ───────────────────────


class StructureView(Static):
    """Decomposes token payload into bands, nodes, operators, signals."""

    pulse_phase = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._tokens: list[dict] = []
        self._payload: dict = {}

    def on_mount(self) -> None:
        self.set_interval(0.5, self._toggle_pulse)

    def _toggle_pulse(self) -> None:
        self.pulse_phase = not self.pulse_phase

    def update_payload(self, data: dict) -> None:
        self._tokens = data.get("tokens", [])
        self._payload = data
        self.refresh()

    def watch_pulse_phase(self) -> None:
        if any(t.get("pulse") for t in self._tokens):
            self.refresh()

    def render(self) -> str:
        if not self._tokens:
            return "[dim]awaiting signal...[/]"

        bright = self.pulse_phase
        lines = []

        # Header
        src = self._payload.get("from", "?")
        at = self._payload.get("at", 0)
        ts = time.strftime("%H:%M:%S", time.localtime(at / 1000)) if at else "?"
        count = len(self._tokens)
        pulse_count = sum(1 for t in self._tokens if t.get("pulse"))
        lines.append(f"╔══ SIGNAL ══════════════════════════════╗")
        lines.append(f"║ from: {src:<10} at: {ts}  tokens: {count:>3} ║")
        lines.append(f"║ pulsing: {pulse_count:<3}  static: {count - pulse_count:<3}             ║")
        lines.append(f"╚════════════════════════════════════════╝")
        lines.append("")

        # Band decomposition
        bands: dict[str, list[dict]] = {}
        for t in self._tokens:
            b = band_name(t["r"])
            bands.setdefault(b, []).append(t)

        band_order = ["apex", "identity", "growth", "power", "seam",
                       "action", "gate", "agents", "alliance", "logic",
                       "operators", "flow"]
        seen_bands = set()

        for bname in band_order:
            if bname not in bands or bname in seen_bands:
                continue
            seen_bands.add(bname)
            toks = bands[bname]
            lines.append(f"┌─ {bname.upper()} ─{'─' * (36 - len(bname))}┐")

            for t in sorted(toks, key=lambda x: (x["r"], x["c"])):
                label = t["label"]
                r, c = t["r"], t["c"]
                cls = classify(label)
                pulse_mark = "◆" if t.get("pulse") else "◇"

                if t.get("pulse") and not bright:
                    pulse_mark = "◈"

                coord = f"[{r:>2},{c:>2}]"
                lines.append(f"│ {pulse_mark} {coord} {label:<8} {cls:<6} │")

            lines.append(f"└{'─' * 40}┘")
            lines.append("")

        # Node graph
        nodes = [t for t in self._tokens if classify(t["label"]) == "node"]
        ops = [t for t in self._tokens if classify(t["label"]) == "op"]
        if nodes:
            lines.append("┌─ GRAPH ────────────────────────────────┐")
            for n in nodes:
                lines.append(f"│ ◉ {n['label']:<12} @ [{n['r']:>2},{n['c']:>2}]             │")
            if ops:
                lines.append("│                                        │")
                lines.append("│ edges:                                 │")
                for o in ops:
                    lines.append(f"│   {o['label']}  [{o['r']:>2},{o['c']:>2}]                          │")
            lines.append(f"└{'─' * 40}┘")
            lines.append("")

        # Signal density
        lines.append("┌─ DENSITY ──────────────────────────────┐")
        row_density = [0] * GRID_SIZE
        col_density = [0] * GRID_SIZE
        for t in self._tokens:
            row_density[t["r"]] += 1
            col_density[t["c"]] += 1
        hot_rows = sorted(range(GRID_SIZE), key=lambda i: -row_density[i])[:5]
        hot_cols = sorted(range(GRID_SIZE), key=lambda i: -col_density[i])[:5]
        hr = " ".join(f"r{r}:{row_density[r]}" for r in hot_rows if row_density[r])
        hc = " ".join(f"c{c}:{col_density[c]}" for c in hot_cols if col_density[c])
        lines.append(f"│ rows: {hr:<33}│")
        lines.append(f"│ cols: {hc:<33}│")
        lines.append(f"└{'─' * 40}┘")

        return "\n".join(lines)


# ── Signal Trace — compact coordinate view ───────────────────────


class SignalTrace(Static):
    """Minimap: sparse coordinate trace of active tokens."""

    pulse_phase = reactive(False)

    def __init__(self) -> None:
        super().__init__()
        self._tokens: list[dict] = []

    def on_mount(self) -> None:
        self.set_interval(0.5, self._toggle)

    def _toggle(self) -> None:
        self.pulse_phase = not self.pulse_phase

    def update_tokens(self, tokens: list[dict]) -> None:
        self._tokens = tokens
        self.refresh()

    def watch_pulse_phase(self) -> None:
        self.refresh()

    def render(self) -> str:
        if not self._tokens:
            return ""
        bright = self.pulse_phase
        # Render a compact 32-row trace: each row shows active columns
        lines = []
        by_row: dict[int, list[dict]] = {}
        for t in self._tokens:
            by_row.setdefault(t["r"], []).append(t)

        for r in range(GRID_SIZE):
            if r not in by_row:
                lines.append(f"[dim]{r:>2}│{'·' * 32}│[/]")
                continue
            row_chars = list("·" * 32)
            for t in by_row[r]:
                c = min(t["c"], 31)
                if t.get("pulse") and not bright:
                    row_chars[c] = "◈"
                elif t.get("pulse"):
                    row_chars[c] = "◆"
                else:
                    row_chars[c] = "■"
            lines.append(f"{r:>2}│{''.join(row_chars)}│")
        return "\n".join(lines)


# ── Gloss Log ────────────────────────────────────────────────────


class GlossLog(RichLog):
    """Scrolling log of bridge messages — machine format."""

    def append_payload(self, data: dict) -> None:
        ts = time.strftime("%H:%M:%S")
        src = data.get("from", "?")
        gloss = data.get("gloss", "")
        count = len(data.get("tokens", []))
        pulse = sum(1 for t in data.get("tokens", []) if t.get("pulse"))
        nodes = [t["label"] for t in data.get("tokens", []) if classify(t["label"]) == "node"]
        node_str = ",".join(nodes) if nodes else "∅"

        self.write(f"[dim]{ts}[/] [{src}] {count}t {pulse}p nodes=[{node_str}]")
        if gloss:
            self.write(f"       [dim italic]{gloss}[/]")


# ── Compose Bar ──────────────────────────────────────────────────


class ComposeBar(Horizontal):
    """Input area for sending tokens to the bridge inbox."""

    def compose(self) -> ComposeResult:
        presets = [("-- preset --", "")]
        if PRESETS_DIR.is_dir():
            for p in sorted(PRESETS_DIR.glob("*.json")):
                presets.append((p.stem, str(p)))
        yield Select(presets, id="preset-select", prompt="preset")
        yield Input(placeholder="r,c,label; r,c,label // gloss", id="compose-input")


# ── Main App ─────────────────────────────────────────────────────


class TokenspeakTUI(App):
    """⟟ tokenspeak — machine window into the ∅ bridge."""

    TITLE = "⟟ tokenspeak · machine"
    CSS = """
    Screen { background: #0a0a0a; }
    Header { background: #0a0a0a; color: #33ff00; }
    Footer { background: #12120a; color: #33ff00; }

    #main { height: 1fr; }
    #left { width: 44; }

    StructureView {
        width: 1fr;
        border: heavy #1f521f;
        color: #33ff00;
        background: #0a0a0a;
        padding: 0 1;
        overflow-y: auto;
    }

    SignalTrace {
        width: 36;
        height: 34;
        border: heavy #1a3a1a;
        color: #1aff1a;
        background: #050505;
        padding: 0;
    }

    GlossLog {
        width: 1fr;
        border: heavy #1f521f;
        background: #12120a;
        color: #33ff00;
    }

    ComposeBar {
        dock: bottom;
        height: 3;
        background: #12120a;
        border-top: heavy #1f521f;
    }

    ComposeBar Select {
        width: 18;
    }

    ComposeBar Input {
        width: 1fr;
        background: #0a0a0a;
        color: #33ff00;
    }

    #status {
        dock: top;
        height: 1;
        background: #0a0a0a;
        color: #33ff00;
        text-align: center;
    }
    """

    BINDINGS = [
        Binding("ctrl+s", "send", "Send"),
        Binding("ctrl+p", "focus_preset", "Preset"),
        Binding("ctrl+l", "clear_log", "Clear"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self, url: str = DEFAULT_URL) -> None:
        super().__init__()
        self.url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=3.0)
        self._last_at = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="status")
        with Horizontal(id="main"):
            with Vertical(id="left"):
                yield SignalTrace()
                yield GlossLog(max_lines=200)
            yield StructureView()
        yield ComposeBar()
        yield Footer()

    def on_mount(self) -> None:
        self._update_status("connecting", "yellow")
        self.set_interval(1.0, self._poll)

    def _update_status(self, state: str, color: str) -> None:
        indicator = f"[{color}]●[/] {state} · {self.url}"
        try:
            self.query_one("#status", Static).update(indicator)
        except Exception:
            pass

    @work(exclusive=True)
    async def _poll(self) -> None:
        try:
            resp = await self._client.get(f"{self.url}/bridge/outbox")
            data = resp.json()
            at = data.get("at")
            if at != self._last_at:
                self._last_at = at
                tokens = data.get("tokens", [])
                self.query_one(StructureView).update_payload(data)
                self.query_one(SignalTrace).update_tokens(tokens)
                self.query_one(GlossLog).append_payload(data)
            self._update_status("live", "#33ff00")
        except Exception:
            self._update_status("offline", "red")

    @work(exclusive=True)
    async def _send(self, payload: dict) -> None:
        payload.setdefault("from", "tui")
        payload.setdefault("at", int(time.time() * 1000))
        try:
            resp = await self._client.post(
                f"{self.url}/bridge/inbox",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code == 200:
                self.query_one(GlossLog).append_payload(payload)
                self._update_status("sent", "#33ff00")
            else:
                self._update_status(f"error {resp.status_code}", "red")
        except Exception as e:
            self._update_status(f"send failed: {e}", "red")

    def _parse_quick(self, text: str) -> dict:
        """Parse quick format: r,c,label [pulse]; ... // gloss"""
        gloss = ""
        if "//" in text:
            text, gloss = text.split("//", 1)
            gloss = gloss.strip()
        tokens = []
        for part in text.split(";"):
            part = part.strip()
            if not part:
                continue
            bits = part.split(",", 2)
            if len(bits) < 3:
                continue
            r, c = int(bits[0].strip()), int(bits[1].strip())
            rest = bits[2].strip()
            pulse = 0
            if rest.endswith(" pulse"):
                pulse = 1
                rest = rest[:-6].strip()
            tok = {"r": r, "c": c, "label": rest}
            if pulse:
                tok["pulse"] = 1
            tokens.append(tok)
        return {"tokens": tokens, "gloss": gloss}

    def action_send(self) -> None:
        inp = self.query_one("#compose-input", Input)
        text = inp.value.strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = self._parse_quick(text)
        if payload.get("tokens"):
            self._send(payload)
            inp.value = ""

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "preset-select":
            return
        path = event.value
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            tokens = data.get("tokens", [])
            gloss = data.get("gloss", data.get("name", ""))
            src = data.get("from", "tui")
            payload = {"tokens": tokens, "gloss": gloss, "from": src}
            self._send(payload)
        except Exception:
            pass

    def action_focus_preset(self) -> None:
        self.query_one("#preset-select", Select).focus()

    def action_clear_log(self) -> None:
        self.query_one(GlossLog).clear()

    async def on_unmount(self) -> None:
        await self._client.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="⟟ tokenspeak · machine window")
    parser.add_argument("--url", default=DEFAULT_URL, help="Bridge URL")
    args = parser.parse_args()
    TokenspeakTUI(url=args.url).run()
