#!/usr/bin/env python3
"""TTFT by uncached prompt length (profiling phase) from aiperf's profile_export.jsonl: does the prefill
scheduling policy (shortest-prefill-first) starve long prompts? Uncached = input - prompt cache-read tokens
when the server reports them. Usage: ttft_by_len.py PROFILE_EXPORT.jsonl"""
import json, sys, statistics as st

def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")

rows = []
for line in open(sys.argv[1]):
    r = json.loads(line)
    if r["metadata"].get("benchmark_phase") != "profiling" or r["metadata"].get("was_cancelled"):
        continue
    m = r["metrics"]
    if "time_to_first_token" not in m:
        continue
    isl = m["input_sequence_length"]["value"]
    cached = m.get("usage_prompt_cache_read_tokens", {}).get("value", 0) or 0
    rows.append((isl - cached, isl, cached, m["time_to_first_token"]["value"] / 1e3, r["metadata"].get("turn_index", 0)))
print(f"requests {len(rows)}; with cache-read reported {sum(1 for r in rows if r[2] > 0)}")
bins = [0, 1024, 4096, 16384, 32768, 65536, 131072, 10**9]
key = 0 if any(r[2] > 0 for r in rows) else 1
print("by", "uncached tokens" if key == 0 else "input tokens (no cache-read reported)")
print(f"{'bin':>16s} {'n':>6s} {'TTFT p50':>9s} {'p90':>7s} {'p99':>7s} {'max':>7s}")
for lo, hi in zip(bins, bins[1:]):
    v = [r[3] for r in rows if lo <= r[key] < hi]
    if v:
        print(f"{lo:>7d}-{hi if hi < 10**9 else 'inf':>8} {len(v):6d} {pct(v, .5):9.2f} {pct(v, .9):7.2f} {pct(v, .99):7.2f} {max(v):7.1f}")
long = sorted(rows, key=lambda r: -r[3])[:5]
print("slowest TTFT:", [(round(r[3], 1), r[0], r[1], r[4]) for r in long])
