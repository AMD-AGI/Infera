#!/usr/bin/env python3
"""Append timestamped ionic HCA snapshots until stopped or duration expires."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

from capture_hca_counters import snapshot


running = True


def stop(_signum, _frame) -> None:
    global running
    running = False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--duration", type=float, default=0.0)
    args = parser.parse_args()
    if args.interval <= 0 or args.duration < 0:
        parser.error("interval must be positive and duration non-negative")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    deadline = time.monotonic() + args.duration if args.duration else None
    with args.output.open("a", encoding="utf-8", buffering=1) as stream:
        while running and (deadline is None or time.monotonic() < deadline):
            started = time.monotonic()
            try:
                record = snapshot()
            except Exception as exc:  # Preserve monitoring gaps in-band.
                record = {"error": f"{type(exc).__name__}: {exc}", "monotonic": started}
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            delay = args.interval - (time.monotonic() - started)
            if delay > 0 and running:
                time.sleep(delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
