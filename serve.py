#!/usr/bin/env python3
"""∅ bridge server — serves MYTHOSGIFT + handles bridge IPC over localhost.

Routes:
  GET  /               → MYTHOSGIFT (the grid)
  GET  /bridge/outbox  → latest outbox.json (CLI → browser)
  POST /bridge/inbox   → write inbox.json (browser → CLI)
  POST /status         → unified lattice status (token-gated)
"""

import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger("tokenspeak-bridge")

# Import JARVIS device input collectors
_JARVIS_PATH = Path.home() / "Obsidian-Vault" / "projects" / "jarvis" / "jarvis"
if _JARVIS_PATH.is_dir() and str(_JARVIS_PATH) not in sys.path:
    sys.path.insert(0, str(_JARVIS_PATH))
from inputs import collect_battery, collect_thermals, collect_network, collect_uptime

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7777
GIFT = Path.home() / "MYTHOSGIFT"
BRIDGE = Path.home() / "lattice" / "seam" / "tokenspeak-bridge"
OUTBOX = BRIDGE / "outbox.json"
INBOX = BRIDGE / "inbox.json"
QUEUE = Path.home() / "lattice" / "seam" / "command-queue"
NODE_AUTH = QUEUE / ".node-auth"

MAX_BODY = 64 * 1024  # 64KB max payload
MAX_TOKENS = 64
ALLOWED_ORIGINS = {f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"}

RAM_WARN_BYTES = 900 * 1024 * 1024  # 900MB
RAM_CHECK_INTERVAL = 5  # seconds
CACHE_TTL = 10  # status cache seconds

# Rate limiting: 30 req/min per IP
RATE_LIMIT = 30
RATE_WINDOW = 60
_rate_buckets = defaultdict(list)
_rate_lock = threading.Lock()

# ── Distress payload ────────────────────────────────────────────

DISTRESS_PAYLOAD = {
    "tokens": [
        {"r": 0,  "c": 8,  "label": "⛔", "pulse": 1},
        {"r": 0,  "c": 14, "label": "⛔", "pulse": 1},
        {"r": 0,  "c": 20, "label": "⛔", "pulse": 1},
        {"r": 8,  "c": 8,  "label": "🧠", "pulse": 1},
        {"r": 8,  "c": 14, "label": "▓",  "pulse": 1},
        {"r": 8,  "c": 20, "label": "▓",  "pulse": 1},
        {"r": 10, "c": 8,  "label": "↑",  "pulse": 1},
        {"r": 10, "c": 14, "label": "↑",  "pulse": 1},
        {"r": 10, "c": 20, "label": "↑",  "pulse": 1},
        {"r": 15, "c": 14, "label": "⚠",  "pulse": 1},
        {"r": 26, "c": 8,  "label": "¬ok"},
        {"r": 26, "c": 20, "label": "OOM"},
        {"r": 28, "c": 14, "label": "∴"},
        {"r": 31, "c": 6,  "label": "⟲",  "pulse": 1},
        {"r": 31, "c": 14, "label": "⛔", "pulse": 1},
        {"r": 31, "c": 22, "label": "↯",  "pulse": 1},
    ],
    "gloss": "DISTRESS: memory overload · bridge OOM · restart needed",
    "from": "bridge",
}


# ── Helpers ──────────────────────────────────────────────────────


def _atomic_write(path, data):
    """Write data to path atomically via tempfile + rename."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with open(fd, "w") as f:
            f.write(data)
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _get_vmrss():
    """Read VmRSS from /proc/self/status (bytes). Returns 0 on failure."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024  # kB → bytes
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _ram_watchdog():
    """Daemon thread: check RSS every N seconds, emit distress at threshold."""
    while True:
        time.sleep(RAM_CHECK_INTERVAL)
        rss = _get_vmrss()
        if rss >= RAM_WARN_BYTES:
            log.warning("RAM watchdog: VmRSS=%dMB >= %dMB, emitting distress",
                        rss // (1024 * 1024), RAM_WARN_BYTES // (1024 * 1024))
            payload = dict(DISTRESS_PAYLOAD, at=int(time.time() * 1000))
            _atomic_write(OUTBOX, json.dumps(payload, indent=2) + "\n")


def _check_rate(ip):
    """Return True if request is within rate limit, False if exceeded."""
    now = time.time()
    with _rate_lock:
        bucket = _rate_buckets[ip]
        # Prune old entries
        _rate_buckets[ip] = bucket = [t for t in bucket if now - t < RATE_WINDOW]
        if len(bucket) >= RATE_LIMIT:
            return False
        bucket.append(now)
        return True


# ── Status helpers ───────────────────────────────────────────────

_status_cache = {"data": None, "expires": 0}


def _run_cmd(cmd, timeout=2.0):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return -1, ""


def _service_active(name, user=True):
    """Check if a systemd service is active."""
    cmd = ["systemctl", "is-active", name]
    if user:
        cmd.insert(1, "--user")
    rc, out = _run_cmd(cmd)
    return out == "active"


def _check_ollama():
    """Ping ollama API to see if it's responding."""
    rc, out = _run_cmd(["curl", "-sf", "http://localhost:11434/api/tags"])
    return rc == 0


def _file_age(path):
    """Return human-readable age of a file, or 'missing'."""
    try:
        mtime = path.stat().st_mtime
        delta = time.time() - mtime
        if delta < 60:
            return f"{int(delta)}s"
        if delta < 3600:
            return f"{int(delta / 60)}m"
        return f"{int(delta / 3600)}h"
    except OSError:
        return "missing"


def _queue_count(directory):
    """Count JSON files in a queue directory."""
    try:
        return len(list(directory.glob("*.json")))
    except OSError:
        return 0


def _read_auth_token():
    """Read the node-auth file content for comparison."""
    try:
        return NODE_AUTH.read_text().strip()
    except OSError:
        return None


def _verify_auth(body):
    """Verify auth token from request body. Returns (ok, code, msg)."""
    auth = body.get("auth")
    if not auth or not isinstance(auth, dict):
        return False, 401, "missing auth"
    token = auth.get("token")
    if not token:
        return False, 401, "missing token"
    expected = _read_auth_token()
    if expected is None:
        return False, 500, "auth file unreadable"
    # Normalize both sides to canonical JSON for comparison
    if not isinstance(token, str):
        token = json.dumps(token, separators=(",", ":"), sort_keys=True)
    try:
        expected = json.dumps(
            json.loads(expected), separators=(",", ":"), sort_keys=True
        )
    except json.JSONDecodeError:
        pass
    if token != expected:
        return False, 403, "bad token"
    return True, 200, "ok"


def collect_status():
    """Build the full status payload (cached)."""
    now = time.time()
    if _status_cache["data"] is not None and now < _status_cache["expires"]:
        return _status_cache["data"]

    battery = asdict(collect_battery())
    thermals = [asdict(z) for z in collect_thermals()]
    network = asdict(collect_network())
    uptime = asdict(collect_uptime())

    ollama_up = _check_ollama()

    result = {
        "mythos": "online",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "nodes": {
            "null": {
                "status": "online" if ollama_up else "offline",
                "model": "llama3.1:8b",
            },
            "mythos": {"status": "online", "role": "guest"},
            "jarvis": {
                "status": "online" if _service_active("jarvis") else "offline",
            },
        },
        "services": {
            "tokenspeak-bridge": {"active": True, "port": PORT},
            "lattice-power": {"active": _service_active("lattice-power")},
            "sion-ear": {"active": _service_active("sion-ear")},
            "jarvis": {"active": _service_active("jarvis")},
            "ollama": {"active": _service_active("ollama", user=False)},
        },
        "inputs": {
            "voice": {
                "enabled": _service_active("sion-ear"),
                "source": "sion-ear",
            },
            "keyboard": {"enabled": True, "source": "pynput+textual"},
            "sensors": {
                "battery": battery,
                "thermals": thermals,
                "network": network,
                "uptime": uptime,
            },
            "ipc": {
                "queue_inbox": _queue_count(QUEUE / "inbox"),
                "queue_outbox": _queue_count(QUEUE / "outbox"),
            },
            "bridge": {
                "inbox_age": _file_age(INBOX),
                "outbox_age": _file_age(OUTBOX),
            },
        },
    }
    _status_cache["data"] = result
    _status_cache["expires"] = now + CACHE_TTL
    return result


def validate_payload(msg):
    """Reject anything that isn't a valid tokenspeak payload."""
    if not isinstance(msg, dict):
        raise ValueError("not object")
    tokens = msg.get("tokens")
    if not isinstance(tokens, list) or len(tokens) > MAX_TOKENS:
        raise ValueError(f"bad tokens (len={len(tokens) if isinstance(tokens, list) else '?'})")
    for t in tokens:
        if not isinstance(t, dict):
            raise ValueError("token not object")
        r, c = t.get("r"), t.get("c")
        if not (isinstance(r, int) and 0 <= r <= 31):
            raise ValueError(f"bad row {r}")
        if not (isinstance(c, int) and 0 <= c <= 31):
            raise ValueError(f"bad col {c}")
        label = t.get("label")
        if not isinstance(label, str) or not label or len(label) > 8:
            raise ValueError("bad label")
    # Whitelist top-level keys
    allowed_keys = {"tokens", "gloss", "sourceText", "paceMs", "at", "from"}
    for k in msg:
        if k not in allowed_keys:
            raise ValueError(f"unexpected key: {k}")
    return msg


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        log.info(fmt, *args)

    def _cors_headers(self):
        """Add CORS headers to every response."""
        origin = self.headers.get("Origin", "")
        if origin in ALLOWED_ORIGINS:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _check_origin(self):
        origin = self.headers.get("Origin", "")
        if origin and origin not in ALLOWED_ORIGINS:
            self._respond(403, b'{"error":"forbidden origin"}')
            return False
        return True

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self._cors_headers()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._serve_file(GIFT, "text/html; charset=utf-8")
        elif self.path.startswith("/bridge/outbox"):
            self._serve_file(OUTBOX, "application/json")
        else:
            self._respond(404, b'{"error":"not found"}')

    def do_POST(self):
        if not self._check_origin():
            return
        if self.path == "/bridge/inbox":
            # Rate limit
            client_ip = self.client_address[0]
            if not _check_rate(client_ip):
                self._respond(429, b'{"error":"rate limit exceeded"}')
                return
            # Content-Type check
            ct = self.headers.get("Content-Type", "")
            if not ct.startswith("application/json"):
                self._respond(415, b'{"error":"expected application/json"}')
                return
            # Content-Length parse
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                self._respond(400, b'{"error":"bad content-length"}')
                return
            if length > MAX_BODY:
                self._respond(413, b'{"error":"too large"}')
                return
            body = self.rfile.read(length)
            try:
                msg = validate_payload(json.loads(body))
                _atomic_write(INBOX, json.dumps(msg, indent=2) + "\n")
                self._respond(200, b'{"ok":true}')
            except (json.JSONDecodeError, ValueError) as e:
                self._respond(400, json.dumps({"error": str(e)}).encode())
        elif self.path == "/status":
            # Content-Length parse
            try:
                length = int(self.headers.get("Content-Length", 0))
            except (ValueError, TypeError):
                self._respond(400, b'{"error":"bad content-length"}')
                return
            if length > MAX_BODY:
                self._respond(413, b'{"error":"too large"}')
                return
            body = self.rfile.read(length)
            try:
                msg = json.loads(body) if body else {}
            except json.JSONDecodeError:
                self._respond(400, b'{"error":"bad json"}')
                return
            ok, code, reason = _verify_auth(msg)
            if not ok:
                self._respond(code, json.dumps({"error": reason}).encode())
                return
            status = collect_status()
            self._respond(200, json.dumps(status).encode())
        else:
            self._respond(404, b'{"error":"not found"}')

    def _serve_file(self, path, content_type):
        if not path.exists():
            self._respond(404, b'{"error":"file not found"}')
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(data))
        self.send_header("Cache-Control", "no-cache")
        self._cors_headers()
        self.end_headers()
        self.wfile.write(data)

    def _respond(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    # Start RAM watchdog daemon thread
    watchdog = threading.Thread(target=_ram_watchdog, daemon=True)
    watchdog.start()
    log.info("RAM watchdog started (threshold=%dMB)", RAM_WARN_BYTES // (1024 * 1024))

    srv = HTTPServer(("127.0.0.1", PORT), Handler)
    log.info("∅ bridge · http://127.0.0.1:%d", PORT)
    srv.serve_forever()
