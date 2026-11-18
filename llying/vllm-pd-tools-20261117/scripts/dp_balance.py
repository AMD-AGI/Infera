#!/usr/bin/env python3
"""Decode DP-rank balance over the measurement from metrics_sampler samples: per sample, running requests and KV usage
of each rank; prints the distribution of max/mean running, max KV usage and how often the busiest rank is KV-full.
  dp_balance.py RUN_DIR [DECODE_URL_SUFFIX=27301]"""
import json, re, sys, datetime, os
run = sys.argv[1].rstrip("/"); sfx = sys.argv[2] if len(sys.argv) > 2 else "27301"
log = open(f"{run}/agentx-c104/benchmark.log", errors="replace").read()
m = re.search(r"(\d\d):(\d\d):(\d\d)\.\d+ NOTICE\s+Phase profiling \(profiling\) started", log)
day = datetime.datetime.fromtimestamp(os.path.getmtime(f"{run}/agentx-c104/benchmark.log"), datetime.timezone.utc).date()
t0 = datetime.datetime(day.year, day.month, day.day, int(m[1]), int(m[2]), int(m[3]), tzinfo=datetime.timezone.utc).timestamp() + 300
rows = []
for line in open(f"{run}/samples.jsonl"):
    d = json.loads(line)
    if d["t"] < t0 or not d["url"].endswith(sfx) or not d.get("m"):
        continue
    ranks = {k: v for k, v in d["m"].items() if k != "-"}
    run_ = [v.get("vllm:num_requests_running", 0) for v in ranks.values()]
    kv = [v.get("vllm:kv_cache_usage_perc", 0) for v in ranks.values()]
    wt = [v.get("vllm:num_requests_waiting", 0) for v in ranks.values()]
    if len(run_) == 4 and sum(run_) > 0:
        rows.append((max(run_), sum(run_) / 4, max(kv), min(kv), sum(wt), sorted(run_)))
q = lambda xs, p: sorted(xs)[int(p * (len(xs) - 1))]
ratio = [a / b for a, b, *_ in rows if b > 0]
print(f"{run}: {len(rows)} samples from minute 5")
print(f"  busiest-rank running p50 {q([r[0] for r in rows], .5)} p90 {q([r[0] for r in rows], .9)}; mean per rank p50 {q([r[1] for r in rows], .5):.1f}")
print(f"  busiest / mean running p50 {q(ratio, .5):.2f} p90 {q(ratio, .9):.2f}")
print(f"  max KV usage p50 {q([r[2] for r in rows], .5):.2f} p90 {q([r[2] for r in rows], .9):.2f}; min KV usage p50 {q([r[3] for r in rows], .5):.2f}")
full = [r for r in rows if r[2] > 0.95]
print(f"  samples with some rank > 95% KV: {len(full) / len(rows):.0%}; waiting total p50 {q([r[4] for r in rows], .5)} p90 {q([r[4] for r in rows], .9)}")
