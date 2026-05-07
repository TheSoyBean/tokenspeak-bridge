#!/usr/bin/env python3
"""∅ sync — pulls browser emissions from Firefox's localStorage into inbox.json.

For file:// origins, Firefox stores localStorage in a SQLite DB.
This script polls that DB and copies tokenspeak.bridge.inbox into the bridge inbox file.

Usage: python3 sync.py        (runs until killed)
       python3 sync.py once   (single read, then exit)
"""

import json
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

BRIDGE = Path.home() / "lattice" / "seam" / "tokenspeak-bridge"
INBOX = BRIDGE / "inbox.json"

# Firefox localStorage for file:// origins
FF_PROFILE_ROOT = Path.home() / ".mozilla" / "firefox"


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


def find_storage_db():
    """Find the webappsstore.sqlite in the default Firefox profile."""
    profiles_ini = FF_PROFILE_ROOT / "profiles.ini"
    if not profiles_ini.exists():
        return None
    # Find default profile
    import configparser
    config = configparser.ConfigParser()
    config.read(profiles_ini)
    for section in config.sections():
        if config.get(section, "Default", fallback="") == "1" or \
           config.get(section, "Name", fallback="") == "default-release":
            path = config.get(section, "Path", fallback="")
            is_relative = config.get(section, "IsRelative", fallback="1") == "1"
            if is_relative:
                profile_dir = FF_PROFILE_ROOT / path
            else:
                profile_dir = Path(path)
            db = profile_dir / "webappsstore.sqlite"
            if db.exists():
                return db
    # Fallback: glob for it
    for db in FF_PROFILE_ROOT.glob("*/webappsstore.sqlite"):
        return db
    return None

def read_localstorage(db_path, key="tokenspeak.bridge.inbox"):
    """Read a localStorage value for file:// origin."""
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            cursor = conn.execute(
                "SELECT value FROM webappsstore2 WHERE key = ? AND originAttributes = '' AND scope LIKE '%.:file'",
                (key,)
            )
            row = cursor.fetchone()
            return row[0] if row else None
    except (sqlite3.Error, OSError):
        return None

def sync_once(db_path, last_at):
    value = read_localstorage(db_path)
    if not value:
        return last_at
    try:
        msg = json.loads(value)
    except json.JSONDecodeError:
        return last_at
    at = msg.get("at", 0)
    if at <= last_at:
        return last_at
    _atomic_write(INBOX, json.dumps(msg, indent=2) + "\n")
    return at

def main():
    once = len(sys.argv) > 1 and sys.argv[1] == "once"
    db_path = find_storage_db()
    if not db_path:
        print("∅ sync: no Firefox localStorage DB found", file=sys.stderr)
        sys.exit(1)
    print(f"∅ sync: watching {db_path}")
    last_at = 0
    while True:
        last_at = sync_once(db_path, last_at)
        if once:
            break
        time.sleep(1)

if __name__ == "__main__":
    main()
