#!/usr/bin/env python3
"""Throughput of an AgentX run per time window of the measurement (profiling phase), from AIPerf's per-request
profile_export.jsonl: completed requests, mean ISL / OSL (server token counts) and (input + output) tok/s/GPU, so a
3600 s run can be compared with 1800 s runs on its first half, and the drift of the request mix over time is visible.
  window_tput.py RUN_DIR_OR_JSONL [WINDOW_S=300] [N_GPUS=8]
Windows are by request end time from the start of the profiling phase (first profiling request start).
"""
import json
import os
import sys

path = sys.argv[1]
if os.path.isdir(path):
    path = os.path.join(path, "agentx-c104", "aiperf_artifacts", "profile_export.jsonl")
win = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
gpus = int(sys.argv[3]) if len(sys.argv) > 3 else 8


def val(m, k):
    v = m.get(k)
    return v.get("value") if isinstance(v, dict) else v


recs, errors = [], 0
with open(path) as f:
    for line in f:
        r = json.loads(line)
        md = r.get("metadata", {})
        if md.get("benchmark_phase") != "profiling":
            continue
        m = r.get("metrics") or {}
        isl, osl = val(m, "usage_prompt_tokens"), val(m, "usage_completion_tokens")
        if r.get("error") or isl is None or osl is None:
            errors += 1
            continue
        recs.append((md["request_start_ns"], md["request_end_ns"], isl, osl))
if not recs:
    sys.exit("no profiling records")
t0 = min(r[0] for r in recs)
t_end = max(r[1] for r in recs)
dur = (t_end - t0) / 1e9
print(f"{path}: {len(recs)} profiled ok, {errors} errors, span {dur:.0f} s")
print(f"{'window s':>12} {'req':>6} {'req/s':>6} {'ISL mean':>9} {'OSL mean':>8} {'tok/s/GPU':>10}")


def row(label, rs, span):
    if not rs:
        return
    tok = sum(r[2] + r[3] for r in rs)
    print(f"{label:>12} {len(rs):6d} {len(rs) / span:6.3f} {sum(r[2] for r in rs) / len(rs):9.0f} "
          f"{sum(r[3] for r in rs) / len(rs):8.0f} {tok / span / gpus:10.0f}")


n = int(dur // win) + 1
for i in range(n):
    lo, hi = i * win, min((i + 1) * win, dur)
    rs = [r for r in recs if lo <= (r[1] - t0) / 1e9 < hi]
    if hi - lo > win / 2:
        row(f"{lo:.0f}-{hi:.0f}", rs, hi - lo)
half = dur / 2
row("1st half", [r for r in recs if (r[1] - t0) / 1e9 < half], half)
row("2nd half", [r for r in recs if (r[1] - t0) / 1e9 >= half], dur - half)
if dur > 2400:
    row("first 1800", [r for r in recs if (r[1] - t0) / 1e9 < 1800], 1800)
    row("300-1800", [r for r in recs if 300 <= (r[1] - t0) / 1e9 < 1800], 1500)
# The first window only holds requests that started in the profiling phase and already finished (short ones): every
# run has the same dip there, so "steady" (from 300 s) separates configurations better.
row("steady 300-", [r for r in recs if (r[1] - t0) / 1e9 >= 300], dur - 300)
row("all", recs, dur)
