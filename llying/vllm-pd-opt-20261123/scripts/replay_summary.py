#!/usr/bin/env python3
"""Per-target summary of decode_replay.py output.
  replay_summary.py REPLAY.jsonl [SKIP_S=300] ...
Requests sent at least SKIP_S after the first send: count, ok / failed / cancelled, latency (send -> done) p50 / p90,
per-output-token latency (latency / osl) p50 / p90, and generated tokens per second over that part of the replay.
"""
import json
import statistics as st
import sys


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


args = sys.argv[1:]
skip = 300.0
if args and args[-1].replace(".", "", 1).isdigit():
    skip = float(args.pop())
for path in args:
    rs = [json.loads(line) for line in open(path) if line.strip()]
    if not rs:
        print(path, "empty")
        continue
    t0 = min(r["t_sent"] for r in rs)
    t_last = max(r["t_sent"] for r in rs)
    w = [r for r in rs if r["t_sent"] >= t0 + skip]
    ok = [r for r in w if r["status"] == 200]
    bad = [r for r in w if r["status"] not in (200, "cancelled")]
    canc = [r for r in w if r["status"] == "cancelled"]
    lat = [r["t_done"] - r["t_sent"] for r in ok]
    tpot = [(r["t_done"] - r["t_sent"]) * 1e3 / max(r["osl"], 1) for r in ok if r["osl"] >= 64]
    span = t_last - (t0 + skip)
    gen = sum(r["completion_tokens"] or 0 for r in ok if r["t_done"] <= t_last)
    print(f"{path}: {len(w)} requests after {skip:.0f} s (span {span:.0f} s); ok {len(ok)}, failed {len(bad)}, cancelled {len(canc)}; "
          f"latency p50 / p90 {q(lat, .5):.1f} / {q(lat, .9):.1f} s; ms per output token (osl >= 64) p50 / p90 "
          f"{q(tpot, .5):.1f} / {q(tpot, .9):.1f}; generated {gen / max(span, 1):.0f} tok/s by the last send")
    if bad:
        print("  failures:", st.mode([str(r["status"]) for r in bad]), "...")
