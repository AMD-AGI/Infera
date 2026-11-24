#!/usr/bin/env python3
"""Where does TTFT go, per request (P4D4 run dir with USAGE_LOG=1)?
  ttft_analyze.py RUN_DIR
Inputs: prefill_usage_matched.jsonl (router's prefill leg per request: isl, cached tokens, leg_ms = prefill until its
first token) and agentx-c104/aiperf_artifacts/profile_export.jsonl (client TTFT). Measurement phase only.
Prints: TTFT = prefill leg + rest (KV pull, decode queue, decode first step); prefill leg vs uncached tokens (the
leg of requests that start on an idle prefill gives the compute rate; leg - compute = queueing); how much of the
queueing small requests spend behind large ones; and a single-server replay of the prefill (arrivals t0, service =
uncached / rate) under FCFS vs shortest-uncached-first, as an estimate of what SPF admission could save.
"""
import heapq
import json
import os
import sys

import numpy as np


def q(a, ps=(50, 90, 99)):
    return " / ".join(f"{np.percentile(a, p):.2f}" for p in ps)


def main():
    d = sys.argv[1]
    pre = [json.loads(l) for l in open(os.path.join(d, "prefill_usage_matched.jsonl"))]
    pre = [r for r in pre if r.get("phase") == "profiling" and r.get("leg_ms") is not None]
    t0 = np.array([r["t0"] for r in pre])
    isl = np.array([r["isl"] for r in pre], dtype=float)
    cached = np.array([r["cached"] for r in pre], dtype=float)
    leg = np.array([r["leg_ms"] for r in pre], dtype=float) / 1000
    miss = np.maximum(isl - cached, 0)
    print(f"{d}: {len(pre)} measured requests with a prefill leg")
    print(f"  prefill leg s p50/p90/p99 {q(leg)}; mean {leg.mean():.2f}")
    print(f"  uncached tokens p50/p90/p99 {q(miss, (50, 90, 99))}; share of ISL {miss.sum() / isl.sum():.3f}")

    ttft = []
    for l in open(os.path.join(d, "agentx-c104/aiperf_artifacts/profile_export.jsonl")):
        r = json.loads(l)
        if r["metadata"].get("benchmark_phase") == "profiling" and "time_to_first_token" in r["metrics"]:
            ttft.append(r["metrics"]["time_to_first_token"]["value"] / 1000)
    ttft = np.array(ttft)
    print(f"  client TTFT s p50/p90/p99 {q(ttft)}; mean {ttft.mean():.2f} (n {len(ttft)}) -> prefill leg is "
          f"{leg.mean() / ttft.mean():.0%} of the mean, the rest (KV pull + decode queue + first decode step) ~{ttft.mean() - leg.mean():.2f} s")

    # compute rate from requests that found the prefill (nearly) idle: lowest leg per uncached token among big ones
    big = miss > 4096
    rate_obs = miss[big] / leg[big]
    rate = float(np.percentile(rate_obs, 90))  # tokens/s when not queued
    comp = miss / rate + 0.05
    queue = np.maximum(leg - comp, 0)
    print(f"  compute rate ~{rate:,.0f} uncached tok/s (p90 of big requests' miss / leg); queueing = leg - miss / rate: "
          f"mean {queue.mean():.2f} s ({queue.mean() / leg.mean():.0%} of the leg), p50/p90 {q(queue, (50, 90))}")
    for lo, hi in ((0, 2048), (2048, 16384), (16384, 65536), (65536, 1e9)):
        m = (miss >= lo) & (miss < hi)
        if m.sum():
            print(f"    uncached [{int(lo):>6}, {int(hi) if hi < 1e9 else 'inf':>6}): n {m.sum():5d}, leg mean {leg[m].mean():.2f} s, "
                  f"queue mean {queue[m].mean():.2f} s, compute mean {comp[m].mean():.2f} s")

    # single-server replay of the prefill: FCFS vs shortest-uncached-first (non-preemptive at request granularity)
    order = np.argsort(t0)
    arr, svc = t0[order] - t0.min(), comp[order]

    def replay(spf):
        t, i, n, wait = 0.0, 0, len(arr), np.zeros(len(arr))
        pq = []
        while i < n or pq:
            if not pq and i < n and arr[i] > t:
                t = arr[i]
            while i < n and arr[i] <= t:
                heapq.heappush(pq, (svc[i] if spf else arr[i], i)); i += 1
            _, j = heapq.heappop(pq)
            wait[j] = t - arr[j]
            t += svc[j]
        return wait

    wf, ws = replay(False), replay(True)
    util = svc.sum() / (arr.max() - arr.min() + svc[-1])
    print(f"  replay (one server at {rate:,.0f} tok/s, utilisation {util:.0%}): mean wait FCFS {wf.mean():.2f} s, SPF {ws.mean():.2f} s "
          f"(p90 {np.percentile(wf, 90):.2f} -> {np.percentile(ws, 90):.2f}); observed queueing mean {queue.mean():.2f} s")


if __name__ == "__main__":
    main()
