#!/usr/bin/env python3
"""Client-side decode step latency from AgentX per-request records.

Each streamed chunk carries one decode step's accepted tokens, so the gaps in
`inter_chunk_latency` are the decode step times seen by the client. The first
gap (first -> second token) spans the prefill -> decode handoff and is skipped.

Usage: chunk_latency.py PROFILE_EXPORT_JSONL [PROFILE_EXPORT_JSONL ...]
Prints one row per file: chunk-gap percentiles, mean, and the share of decode
time spent in gaps above 50 / 100 / 250 ms. Profiling-phase requests only.
"""
import json
import sys


def load(path):
    gaps, reqs = [], 0
    for line in open(path):
        d = json.loads(line)
        if d["metadata"].get("benchmark_phase") != "profiling" or d.get("error"):
            continue
        icl = d["metrics"].get("inter_chunk_latency", {}).get("value")
        if not icl or len(icl) < 3:
            continue
        gaps.extend(icl[1:])
        reqs += 1
    gaps.sort()
    return gaps, reqs


def main():
    print(f"{'arm':>8s} {'reqs':>5s} {'chunks':>8s} {'p10':>6s} {'p50':>6s} {'p90':>6s} {'p99':>6s} "
          f"{'p99.9':>7s} {'max':>7s} {'mean':>6s} | time share in gaps >50ms >100ms >250ms")
    for path in sys.argv[1:]:
        g, reqs = load(path)
        tot = sum(g)
        share = lambda t: 100 * sum(x for x in g if x > t) / tot
        q = [g[min(len(g) - 1, int(p * len(g)))] for p in (.1, .5, .9, .99, .999)]
        arm = path.rstrip("/").split("/")[-3].split("-agentx")[0]
        print(f"{arm:>8s} {reqs:5d} {len(g):8d} {q[0]:6.1f} {q[1]:6.1f} {q[2]:6.1f} {q[3]:6.1f} "
              f"{q[4]:7.1f} {g[-1]:7.0f} {tot / len(g):6.2f} | {share(50):5.1f}% {share(100):5.1f}% {share(250):5.1f}%")


if __name__ == "__main__":
    main()
