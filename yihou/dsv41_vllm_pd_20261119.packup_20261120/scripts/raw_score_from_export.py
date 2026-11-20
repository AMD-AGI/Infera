#!/usr/bin/env python3
"""Recompute AgentX tok/s/GPU from aiperf profile_export.jsonl (profiling-phase, non-error records only).
score = (sum ISL + sum OSL) / (max request_end - min request_start) / gpus.
Validated 2026-11-19: reproduces the harness score of e1-r1-c128-11191033 exactly (255,796.7).
Used to salvage runs cut short by node preemption (window < 3600 s => NOT an official score).
Usage: raw_score_from_export.py <profile_export.jsonl> [gpus=2]"""
import json, sys
p = sys.argv[1]; gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 2
n = isl = osl = err = 0; s = e = None
for line in open(p):
    o = json.loads(line); md = o.get("metadata", {})
    if md.get("benchmark_phase") not in (None, "profiling"):
        continue
    if o.get("error"):
        err += 1; continue
    m = o.get("metrics", {})
    v = lambda k: (m.get(k) or {}).get("value") if isinstance(m.get(k), dict) else m.get(k)
    n += 1; isl += int(v("input_sequence_length") or 0); osl += int(v("output_sequence_length") or 0)
    a, b = md.get("request_start_ns"), md.get("request_end_ns")
    if a: s = a if s is None else min(s, a)
    if b: e = b if e is None else max(e, b)
d = (e - s) / 1e9
print(f"n={n} err={err} window={d:.0f}s tok/s/GPU={(isl+osl)/d/gpus:,.1f} qps={n/d:.3f} ISL_mean={isl/n:,.0f}")
