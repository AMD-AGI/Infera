#!/usr/bin/env python3
"""Ideal prefill computation of an AgentX run vs what the prefill actually computed, without per-request logs.
Ideal: for a request whose conversation had an earlier turn in the run (warmup included), ISL - ISL(previous turn)
(append-only history, previous prompt fully cached); else its whole ISL (cold start; an upper bound, since another
conversation may share the prefix). Actual: prefill counters (prompt - local hits - CPU-tier hits) over the same window,
from metrics_sampler samples.
  ideal_hit.py RUN_DIR [PREFILL_URL_SUFFIX=27201]
"""
import collections
import json
import sys

run = sys.argv[1].rstrip("/")
sfx = sys.argv[2] if len(sys.argv) > 2 else "27201"
recs = []
for line in open(f"{run}/agentx-c104/aiperf_artifacts/profile_export.jsonl"):
    r = json.loads(line)
    md, m = r.get("metadata", {}), r.get("metrics") or {}
    isl = (m.get("usage_prompt_tokens") or {}).get("value")
    if isl is None:
        continue
    recs.append(dict(conv=md.get("conversation_id"), turn=md.get("turn_index"), phase=md.get("benchmark_phase"),
                     depth=md.get("agent_depth"), t0=md["request_start_ns"] / 1e9, t1=md["request_end_ns"] / 1e9,
                     isl=int(isl), osl=int((m.get("usage_completion_tokens") or {}).get("value") or 0)))
by = collections.defaultdict(list)
for r in recs:
    by[r["conv"]].append(r)
for v in by.values():
    v.sort(key=lambda x: (x["turn"], x["t0"]))
    for a, b in zip([None] + v[:-1], v):
        b["prev"] = a
prof = [r for r in recs if r["phase"] == "profiling"]
t0, t1 = min(r["t0"] for r in prof), max(r["t1"] for r in prof)
tot = collections.Counter()
for r in prof:
    a = r["prev"]
    tot["isl"] += r["isl"]
    if a is None or a["isl"] > r["isl"]:
        tot["cold"] += r["isl"]
        tot["n_cold"] += 1
    else:
        tot["warm_new"] += r["isl"] - a["isl"]
        tot["prev_osl"] += min(a["osl"], r["isl"] - a["isl"])
        tot["n_warm"] += 1
ideal = tot["cold"] + tot["warm_new"]
print(f"{run}: {len(prof)} profiled requests ({tot['n_cold']} cold, {tot['n_warm']} warm), window {t1 - t0:.0f} s")
print(f"  ISL {tot['isl']:,}; ideal computed {ideal:,} ({ideal / tot['isl']:.2%}): cold {tot['cold']:,} "
      f"({tot['cold'] / tot['isl']:.2%}), warm new {tot['warm_new']:,} ({tot['warm_new'] / tot['isl']:.2%}, of which "
      f"previous-turn output <= {tot['prev_osl']:,})")
keys = ("vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total", "vllm:external_prefix_cache_hits_total")
pts = []
for line in open(f"{run}/samples.jsonl"):
    d = json.loads(line)
    if not d["url"].endswith(sfx) or not d.get("m"):
        continue
    s = [sum(e.get(k, 0.0) for e in d["m"].values()) for k in keys]
    pts.append((d["t"], s))
pts.sort()
a = min(pts, key=lambda p: abs(p[0] - t0))
b = min(pts, key=lambda p: abs(p[0] - t1))
prompt, local, ext = (b[1][i] - a[1][i] for i in range(3))
comp = prompt - local - ext
print(f"  prefill counters {a[0] - t0:+.0f} .. {b[0] - t1:+.0f} s around the window: prompt {prompt:,.0f} "
      f"(AIPerf ISL {tot['isl']:,}), local {local / prompt:.2%}, CPU tier {ext / prompt:.2%}, computed {comp:,.0f} "
      f"({comp / prompt:.2%}); computed / ideal = {comp / max(ideal, 1):.2f}")
