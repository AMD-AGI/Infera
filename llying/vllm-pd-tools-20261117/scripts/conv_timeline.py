#!/usr/bin/env python3
"""Per-conversation timeline of prefill cache hits from prefill_usage.py's matched records.
  conv_timeline.py RUN_DIR CONV_PREFIX [CONV_PREFIX...] [--last N]
"""
import collections
import json
import sys

run = sys.argv[1].rstrip("/")
last = 14
args = sys.argv[2:]
if "--last" in args:
    i = args.index("--last")
    last = int(args[i + 1])
    args = args[:i] + args[i + 2:]
rs = [json.loads(line) for line in open(f"{run}/prefill_usage_matched.jsonl")]
t00 = min(r["t0"] for r in rs)
by = collections.defaultdict(list)
for r in rs:
    by[r["conv"]].append(r)
for pre in args:
    for conv, v in by.items():
        if not conv.startswith(pre):
            continue
        v.sort(key=lambda r: r["t0"])
        print(f"== {conv[:72]}: {len(v)} requests")
        prev = None
        for r in v[-last:]:
            gap = r["t0"] - prev if prev is not None else 0.0
            prev = r["t0"]
            c = r.get("cached")
            print(f"   +{r['t0'] - t00:7.0f}s gap {gap:6.0f}s {str(r['phase'])[:4]} turn {r['turn']:4d} isl {r['isl']:7d} "
                  f"cached {c if c is not None else -9:7d} leg {r.get('leg_ms')} ms")
