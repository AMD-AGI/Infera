#!/usr/bin/env python3
"""Detect a hung prefill instance: requests queued or in flight, but no "Prefill batch" line for
STALL_SECONDS (default 180). Uses the live engine gauges and the server logs.

Usage: hang_check.py [ROUND] [STALL_SECONDS]   -> prints one line per prefill; exit 1 if any is hung.
"""

import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
ROUND = sys.argv[1] if len(sys.argv) > 1 else "r2"
STALL = int(sys.argv[2]) if len(sys.argv) > 2 else 180
LOGS = KIT / "runtime" / "runs" / ROUND / "launch" / "server-logs"
URLS = {"prefill-0": "http://10.245.148.209:26101", "prefill-1": "http://10.245.153.247:26357",
        "prefill-2": "http://10.245.153.247:26613"}
GAUGE = re.compile(r"^sglang:(num_queue_reqs|num_running_reqs|num_prefill_inflight_queue_reqs)\{[^}]*\} ([0-9.]+)$")


def last_batch(log: Path) -> str:
    with open(log, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - (8 << 20)))
        lines = f.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        if "Prefill batch" in line:
            return line[:26]
    return ""


def main() -> int:
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    hung = False
    for inst, url in URLS.items():
        try:
            text = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
        except Exception as exc:  # noqa: BLE001
            print(f"{inst}: metrics unavailable ({exc})")
            continue
        pending = sum(float(m.group(2)) for m in map(GAUGE.match, text.splitlines()) if m)
        ts = last_batch(LOGS / f"{inst}.log")
        age = (now - datetime.fromisoformat(ts)).total_seconds() if ts else float("inf")
        bad = pending > 0 and age > STALL
        hung |= bad
        print(f"{inst}: pending {pending:.0f}, last batch {age:.0f} s ago{'  <-- HUNG' if bad else ''}")
    return 1 if hung else 0


if __name__ == "__main__":
    sys.exit(main())
