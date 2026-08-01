#!/usr/bin/env python3
"""Pull the four Goal items out of the Case A artifacts.

Goal item 3 (MTP acceptance) is deliberately NOT read from the bench: MTP is off
in this deployment, so the driver's acceptance_length degenerates to ~1 and
reporting it would manufacture a number for a feature that is not running.
"""
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
summary_p = run_dir / "summary.json"
metrics_p = run_dir / "metrics.jsonl"


def pct(vals, q):
    if not vals:
        return None
    s = sorted(vals)
    i = min(int(q * len(s)), len(s) - 1)
    return s[i]


print("=" * 72)
print("GOAL 2 - classic serving metrics (from summary.json)")
print("=" * 72)
if summary_p.exists():
    s = json.loads(summary_p.read_text())
    print(json.dumps(s, indent=2)[:6000])
else:
    print(f"MISSING: {summary_p}")

print()
print("=" * 72)
print("GOAL 4 - session concurrency (num_sessions_active time series)")
print("=" * 72)
if metrics_p.exists():
    active, inflight = [], []
    for line in metrics_p.open():
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("num_sessions_active") is not None:
            active.append(r["num_sessions_active"])
        if r.get("in_flight") is not None:
            inflight.append(r["in_flight"])
    print(f"ticks: {len(active)}")
    if active:
        print(f"  live sessions  p50={pct(active,.5)}  p90={pct(active,.9)}  "
              f"p99={pct(active,.99)}  min={min(active)}  max={max(active)}")
    if inflight:
        print(f"  in-flight      p50={pct(inflight,.5)}  p90={pct(inflight,.9)}  "
              f"p99={pct(inflight,.99)}  max={max(inflight)}")
else:
    print(f"MISSING: {metrics_p}")
