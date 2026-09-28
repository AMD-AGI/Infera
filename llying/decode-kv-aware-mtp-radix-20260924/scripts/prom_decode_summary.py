#!/usr/bin/env python3
"""Decode-side engine metrics per arm, from the before/after /metrics snapshots
run_perf.sh takes around the AgentX step (so warmup and profiling together).

Usage: prom_decode_summary.py EVIDENCE_PERF_DIR ARM [ARM ...]
Prints per-stage request latency means, histogram means and counter deltas,
summed over DP ranks.
"""
import collections
import re
import sys

LINE = re.compile(r"^(sglang:[A-Za-z_]+?)(_sum|_count|_total)?(\{[^}]*\})?\s+([0-9.eE+-]+)$")
STAGE = re.compile(r'stage="([^"]+)"')
HISTS = ["inter_token_latency_seconds", "time_to_first_token_seconds", "kv_transfer_alloc_ms",
         "eviction_duration_seconds", "hicache_backup_duration_seconds", "load_back_duration_seconds",
         "generation_tokens_histogram", "prompt_tokens_histogram"]
COUNTERS = ["evicted_tokens", "hicache_backup_tokens", "load_back_tokens", "hicache_backup_bytes",
            "load_back_bytes", "num_transfer_failed_reqs"]


def load(path):
    d = collections.defaultdict(float)
    for raw in open(path):
        m = LINE.match(raw.strip())
        if not m:
            continue
        name, suffix, labels = m.group(1)[7:], m.group(2) or "", m.group(3) or ""
        st = STAGE.search(labels)
        d[(name + (f"[{st.group(1)}]" if st else ""), suffix)] += float(m.group(4))
    return d


def main():
    base, arms = sys.argv[1], sys.argv[2:]
    delta = {}
    for arm in arms:
        a, b = load(f"{base}/{arm}-decode-before.prom"), load(f"{base}/{arm}-decode-after.prom")
        delta[arm] = {k: v - a.get(k, 0.0) for k, v in b.items()}
    mean = lambda arm, n: (delta[arm].get((n, "_sum"), 0) / delta[arm][(n, "_count")]
                           if delta[arm].get((n, "_count")) else float("nan"))
    head = f"{'metric':50s}" + "".join(f"{a:>14s}" for a in arms)
    print("per-stage request latency, mean ms")
    print(head)
    stages = sorted({k[0] for arm in arms for k in delta[arm] if k[0].startswith("per_stage_req_latency_seconds[")})
    for s in stages:
        print(f"{s[30:-1]:50s}" + "".join(f"{1000 * mean(a, s):14.2f}" for a in arms))
    print("histogram means (native unit)")
    print(head)
    for h in HISTS:
        print(f"{h:50s}" + "".join(f"{mean(a, h):14.5g}" for a in arms))
    print("counter deltas")
    print(head)
    for c in COUNTERS:
        vals = [delta[a].get((c, "_total"), delta[a].get((c, ""), 0.0)) for a in arms]
        print(f"{c:50s}" + "".join(f"{v:14.4g}" for v in vals))


if __name__ == "__main__":
    main()
