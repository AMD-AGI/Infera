#!/usr/bin/env python3
"""Decode DP-rank balance from metrics_sampler.py samples (from minute SKIP of the sampling on): per sample, running
requests and KV usage of each decode rank; prints busiest/mean running, KV usage spread and how often a rank is KV-full.
  dp_balance.py SAMPLES.jsonl [DECODE_URL_SUFFIX=27301] [SKIP_S=1500]"""
import json, sys
path = sys.argv[1]; sfx = sys.argv[2] if len(sys.argv) > 2 else "27301"; skip = float(sys.argv[3]) if len(sys.argv) > 3 else 1500
rows, t0 = [], None
for line in open(path):
    d = json.loads(line)
    t0 = d["t"] if t0 is None else t0
    if d["t"] < t0 + skip or not d["url"].endswith(sfx) or not d.get("m"):
        continue
    ranks = {k: v for k, v in d["m"].items() if k != "-"}
    run_ = [v.get("vllm:num_requests_running", 0) for v in ranks.values()]
    kv = [v.get("vllm:kv_cache_usage_perc", 0) for v in ranks.values()]
    wt = [v.get("vllm:num_requests_waiting", 0) for v in ranks.values()]
    if len(run_) == 4 and sum(run_) > 0:
        rows.append((max(run_), sum(run_) / 4, max(kv), min(kv), sum(wt)))
if not rows:
    sys.exit(f"{path}: no decode samples")
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
ratio = [a / b for a, b, *_ in rows if b > 0]
print(f"{path}: {len(rows)} decode samples after {skip:.0f} s")
print(f"  busiest / mean running p50 {q(ratio, .5):.2f} p90 {q(ratio, .9):.2f}; mean running per rank p50 {q([r[1] for r in rows], .5):.1f}")
print(f"  KV usage max-rank p50 {q([r[2] for r in rows], .5):.2f} p90 {q([r[2] for r in rows], .9):.2f}; min-rank p50 {q([r[3] for r in rows], .5):.2f}")
print(f"  samples with a rank > 95% KV: {sum(r[2] > 0.95 for r in rows) / len(rows):.0%}; waiting (all ranks) p50 {q([r[4] for r in rows], .5)} p90 {q([r[4] for r in rows], .9)}")
