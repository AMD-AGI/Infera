#!/usr/bin/env python3
"""Watch the prefill engines and capture the first stall of each one with hang_capture.sh.

An instance is stalled when it has requests queued, running or in flight while its progress
counters (sglang:realtime_tokens_total + sglang:kv_transfer_latency_ms_count) have not moved
for STALL seconds. Both come from the engine's own /metrics (served by the HTTP process, which
stays up during a scheduler hang), so a dead server-log follower cannot raise a false alarm the
way a log-based check would. Each instance is captured at most once per watcher run.

Usage: hang_watch.py [ROUND=r2] [STALL_SECONDS=600] [INTERVAL_SECONDS=30]
Needs the config environment (BENCH_DIR, TOPOLOGY, ENGINE_PORT_BASE, ENGINE_PORT_STRIDE);
scripts/start_hang_watch.sh provides it and runs the watcher in the background on the control node.
Output: one line per poll and per capture on stdout (runtime/runs/<ROUND>/tune/hang_watch.log).
"""

import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = sys.argv[1] if len(sys.argv) > 1 else "r2"
STALL = int(sys.argv[2]) if len(sys.argv) > 2 else 600
INTERVAL = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SAMPLE = re.compile(r"^(sglang:[a-z_]+)\{[^}]*\} ([0-9.eE+-]+)$")
PROGRESS = {"sglang:realtime_tokens_total", "sglang:kv_transfer_latency_ms_count"}
PENDING = {"sglang:num_queue_reqs", "sglang:num_running_reqs", "sglang:num_prefill_inflight_queue_reqs",
           "sglang:num_prefill_prealloc_queue_reqs", "sglang:num_bootstrap_queue_reqs"}


def prefills() -> dict[str, str]:
    rows = subprocess.run(
        ["python3", f"{os.environ['BENCH_DIR']}/tools/topology.py", "rows", os.environ["TOPOLOGY"]],
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    base, stride = int(os.environ["ENGINE_PORT_BASE"]), int(os.environ.get("ENGINE_PORT_STRIDE") or 1)
    urls = {}
    for row in rows:
        index, instance, role, _node, ip = row.split("\t")
        if role == "prefill":
            urls[instance] = f"http://{ip}:{base + int(index) * stride}/metrics"
    return urls


def read(url: str) -> tuple[float, float]:
    progress = pending = 0.0
    text = urllib.request.urlopen(url, timeout=10).read().decode()
    for line in text.splitlines():
        m = SAMPLE.match(line)
        if not m:
            continue
        if m.group(1) in PROGRESS:
            progress += float(m.group(2))
        elif m.group(1) in PENDING:
            pending += float(m.group(2))
    return progress, pending


def main() -> int:
    urls = prefills()
    last: dict[str, tuple[float, float]] = {}  # instance -> (progress value, time it last changed)
    captured: set[str] = set()
    while True:
        now = time.time()
        parts = []
        for inst, url in urls.items():
            try:
                progress, pending = read(url)
            except Exception as exc:  # noqa: BLE001  (engine down or restarting)
                last.pop(inst, None)
                parts.append(f"{inst} down ({type(exc).__name__})")
                continue
            if inst not in last or progress != last[inst][0] or pending == 0:
                last[inst] = (progress, now)
            stalled = now - last[inst][1]
            parts.append(f"{inst} pending {pending:.0f} stalled {stalled:.0f}s")
            if pending > 0 and stalled >= STALL and inst not in captured:
                captured.add(inst)
                stamp = datetime.now(timezone.utc).strftime("%FT%TZ")
                print(f"{stamp} {inst} STALLED {stalled:.0f}s with {pending:.0f} pending: capturing", flush=True)
                result = subprocess.run(["bash", str(HERE / "hang_capture.sh"), f"INSTANCE={inst}", f"ROUND={ROUND}"],
                                        capture_output=True, text=True)
                print(f"{stamp} {inst} capture rc={result.returncode} {result.stdout.strip()} {result.stderr.strip()[-300:]}",
                      flush=True)
        print(datetime.now(timezone.utc).strftime("%FT%TZ ") + "; ".join(parts), flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    sys.exit(main())
