#!/usr/bin/env python3
"""Little's-law residence per phase from a metrics_sampler TSV over a profiling window.

Usage: residence.yihou.py TSV DATE START_HH:MM:SS END_HH:MM:SS QPS
residence(phase) = mean over samples of (sum over DP ranks of the gauge) / QPS   [seconds per request]
Also prints mean/max of usage gauges (per-DP max).
"""
import sys, collections, datetime as dt

tsv, day, t0s, t1s, qps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
mk = lambda s: dt.datetime.fromisoformat(f"{day}T{s}+00:00").timestamp()
t0, t1 = mk(t0s), mk(t1s)
GAUGES = {("prefill", "num_queue_reqs"): "P queue (wait for prefill)",
          ("prefill", "num_prefill_bootstrap_queue_reqs"): "P bootstrap",
          ("prefill", "num_running_reqs"): "P forward/chunking",
          ("prefill", "num_prefill_inflight_queue_reqs"): "P KV send (inflight)",
          ("decode", "num_decode_prealloc_queue_reqs"): "D prealloc (admission wait)",
          ("decode", "num_decode_transfer_queue_reqs"): "D transfer queue",
          ("decode", "num_queue_reqs"): "D waiting (arrived, not batched)",
          ("decode", "num_running_reqs"): "D running (decode)"}
USAGE = [("prefill", "token_usage"), ("prefill", "cache_hit_rate"), ("decode", "token_usage")]
snap = collections.defaultdict(lambda: collections.defaultdict(float))
use = collections.defaultdict(list)
host = collections.defaultdict(dict)
for ln in open(tsv):
    t, leg, dp, m, v = ln.rstrip("\n").split("\t")
    t = float(t)
    if not (t0 <= t <= t1) or dp == "-":
        continue
    try:
        v = float(v)
    except ValueError:
        continue
    if (leg, m) in GAUGES:
        snap[(leg, m)][t] += v
    if (leg, m) in USAGE:
        use[(leg, m, dp)].append(v)
    if leg == "prefill" and m in ("hicache_host_used_tokens", "hicache_host_total_tokens"):
        host[(dp, m)][t] = v
for k, name in GAUGES.items():
    s = snap[k]
    if s:
        mean = sum(s.values()) / len(s)
        print(f"{name:34s} mean_sum={mean:6.2f}  residence={mean / qps:6.2f} s  max_sum={max(s.values()):.0f}")
for (leg, m) in USAGE:
    vals = {dp: v for (l, mm, dp), v in use.items() if l == leg and mm == m}
    print(f"{leg} {m:15s} mean " + " ".join(f"{dp}:{sum(v)/len(v):.2f}" for dp, v in sorted(vals.items()))
          + " | max " + " ".join(f"{dp}:{max(v):.2f}" for dp, v in sorted(vals.items())))
full = {}
for dp in sorted({d for d, _ in host}):
    u, tot = host[(dp, "hicache_host_used_tokens")], host[(dp, "hicache_host_total_tokens")]
    ts = sorted(t for t in u if t in tot and tot[t] > 0)
    f = [t for t in ts if u[t] / tot[t] >= 0.999]
    full[dp] = f"{(f[0] - t0) / 60:.0f}min" if f else "never"
print("prefill host pool >=99.9% full first at (from window start):", full)
