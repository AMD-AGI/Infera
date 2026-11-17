#!/usr/bin/env python3
"""PD bottleneck numbers over the AgentX measurement window from metrics_sampler.py samples.
  samples_summary.py SAMPLES.jsonl PREFILL_URL DECODE_URL [BENCHMARK_LOG]
The window is AIPerf's "Phase profiling (profiling) started" .. "sending complete" in BENCHMARK_LOG (UTC, same day),
or the whole file. Gauges: median / p90 over samples; counters: rate over the window.
"""
import datetime as dt
import json
import re
import statistics as st
import sys


def window(log):
    if not log:
        return None, None
    t0 = t1 = None
    day = dt.datetime.fromtimestamp(__import__("os").path.getmtime(log), dt.timezone.utc).date()
    for line in open(log, errors="replace"):
        if "Phase profiling (profiling) started" in line:
            t0 = line[:12]
        elif "Phase profiling (profiling) sending complete" in line:
            t1 = line[:12]
    def ts(hms):
        h, m, s = hms.split(":")
        return dt.datetime(day.year, day.month, day.day, int(h), int(m), int(float(s)), tzinfo=dt.timezone.utc).timestamp()
    return (ts(t0) if t0 else None), (ts(t1) if t1 else None)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


def main():
    path, purl, durl = sys.argv[1:4]
    t0, t1 = window(sys.argv[4] if len(sys.argv) > 4 else None)
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "m" in r and (t0 is None or t0 <= r["t"] <= (t1 or 1e18))]
    P = [r for r in rows if r["url"] == purl]
    D = [r for r in rows if r["url"] == durl]
    print(f"window {t0} .. {t1}: {len(P)} prefill / {len(D)} decode samples")

    def g(r, eng, k):
        return r["m"].get(eng, {}).get(k)

    def counter_rate(rs, key):
        if len(rs) < 2:
            return float("nan"), 0.0
        a = sum(v.get(key, 0) for v in rs[0]["m"].values())
        b = sum(v.get(key, 0) for v in rs[-1]["m"].values())
        return (b - a) / max(rs[-1]["t"] - rs[0]["t"], 1), b - a

    # decode
    run_max, run_sum, wait_sum, kv = [], [], [], []
    for r in D:
        ranks = [e for e in r["m"] if e != "-"]
        rr = [g(r, e, "vllm:num_requests_running") or 0 for e in ranks]
        run_max.append(max(rr) if rr else 0)
        run_sum.append(sum(rr))
        wait_sum.append(sum(g(r, e, "vllm:num_requests_waiting") or 0 for e in ranks))
        kv += [g(r, e, "vllm:kv_cache_usage_perc") or 0 for e in ranks]
    gen_rate, _ = counter_rate(D, "vllm:generation_tokens_total")
    _, drafts = counter_rate(D, "vllm:spec_decode_num_drafts_total")
    _, acc = counter_rate(D, "vllm:spec_decode_num_accepted_tokens_total")
    print(f"decode: running sum p50 {q(run_sum, .5):.0f} p90 {q(run_sum, .9):.0f}; busiest rank p50 {q(run_max, .5):.0f} "
          f"p90 {q(run_max, .9):.0f}; waiting sum p50 {q(wait_sum, .5):.0f}; KV usage p50 {q(kv, .5):.2f} p90 {q(kv, .9):.2f}; "
          f"gen {gen_rate:,.0f} tok/s; accept len {1 + acc / drafts if drafts else float('nan'):.2f}")
    # prefill
    pr = [g(r, "0", "vllm:num_requests_running") or 0 for r in P]
    pw = [g(r, "0", "vllm:num_requests_waiting") or 0 for r in P]
    pkv = [g(r, "0", "vllm:kv_cache_usage_perc") or 0 for r in P]
    prompt_rate, _ = counter_rate(P, "vllm:prompt_tokens_total")
    _, hits = counter_rate(P, "vllm:prefix_cache_hits_total")
    _, qs = counter_rate(P, "vllm:prefix_cache_queries_total")
    _, ehits = counter_rate(P, "vllm:external_prefix_cache_hits_total")
    _, eqs = counter_rate(P, "vllm:external_prefix_cache_queries_total")
    req_rate, nreq = counter_rate(P, "vllm:request_success_total")
    busy = sum(1 for w in pw if w > 0) / max(len(pw), 1)
    computed = prompt_rate * (1 - (hits + ehits) / qs) if qs else float("nan")
    print(f"prefill: {req_rate * 60:.0f} req/min ({nreq:.0f}); prompt {prompt_rate:,.0f} tok/s, computed ~{computed:,.0f} tok/s; "
          f"local hit {hits / qs if qs else float('nan'):.3f}, CPU-tier hit {ehits / qs if qs else float('nan'):.3f}; "
          f"queue non-empty {100 * busy:.0f}% (waiting p50 {q(pw, .5):.0f} p90 {q(pw, .9):.0f}); running p50 {q(pr, .5):.0f}; "
          f"KV usage p50 {q(pkv, .5):.2f}")


if __name__ == "__main__":
    main()
