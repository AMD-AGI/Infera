#!/usr/bin/env python3
"""Rebuild an AgentX score row from AIPerf's per-request records (aiperf_artifacts/profile_export.jsonl) when the run
was cut before AgentX wrote agentx_conc*.json (preemption at the end of the window). Profiling-phase, non-cancelled
requests: total / output tokens over the window (first request start .. last request end), per GPU over --gpus
(12 reproduces AgentX's per_gpu figure for the 1P1D PP8 arm: 64c 679,142 tok/s -> 56,595), request count, TTFT,
interactivity ((output tokens - 1) / decode time; its p90 is the slow tail, AgentX's convention) and ITL percentiles. Check it on a run with the official JSON first.
Streams the file (a few floats per record).

Usage: score_from_records.py profile_export.jsonl [--gpus 12]
"""
import argparse
import json
import statistics as st


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))] if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gpus", type=int, default=12)
    a = ap.parse_args()
    t0, t1, tin, tout, n = None, None, 0, 0, 0
    ttft, intv, itl, e2e = [], [], [], []
    for line in open(a.path):
        r = json.loads(line)
        md, m = r["metadata"], r.get("metrics", {})
        if md.get("benchmark_phase") != "profiling" or md.get("was_cancelled") or r.get("error"):
            continue
        if "request_latency" not in m:
            continue
        s, e = md["request_start_ns"], md["request_end_ns"]
        t0 = s if t0 is None else min(t0, s)
        t1 = e if t1 is None else max(t1, e)
        i = m.get("usage_prompt_tokens", m.get("input_sequence_length", {})).get("value", 0)
        o = m.get("output_token_count", m.get("usage_completion_tokens", {})).get("value", 0)
        tin, tout, n = tin + i, tout + o, n + 1
        lat, ft = m["request_latency"]["value"] / 1e3, m.get("time_to_first_token", {}).get("value", 0) / 1e3
        ttft.append(ft)
        e2e.append(lat)
        if o > 1 and lat > ft:
            intv.append((o - 1) / (lat - ft))
            itl.append((lat - ft) / (o - 1))
    dur = (t1 - t0) / 1e9
    print(f"requests {n:,}  window {dur:.1f} s  total {(tin + tout) / dur:,.0f} tok/s  "
          f"per GPU (/{a.gpus}) total {(tin + tout) / dur / a.gpus:,.0f}  output {tout / dur / a.gpus:.1f}  "
          f"req/s {n / dur:.3f}")
    print(f"TTFT p50 / p90 / mean {pct(ttft, .5):.2f} / {pct(ttft, .9):.2f} / {st.mean(ttft):.2f} s  "
          f"intvty p50 / p90 {pct(intv, .5):.1f} / {pct(intv, .1):.1f}  ITL p50 / p90 {pct(itl, .5) * 1e3:.1f} / "
          f"{pct(itl, .9) * 1e3:.1f} ms  E2E p90 {pct(e2e, .9):.1f} s")


if __name__ == "__main__":
    main()
