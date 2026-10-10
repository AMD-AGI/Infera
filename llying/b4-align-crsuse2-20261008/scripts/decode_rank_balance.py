#!/usr/bin/env python3
"""Per-D-rank load over a run's measured window (P8D4 notes, sections 6-7).

Usage: decode_rank_balance.py RUN_DIR   (needs RUN/analysis/summary.json, from analyze.py)

Per rank: requests, input tokens, requests blocked by the KV budget, alloc wait,
mean active KV (used / max_total_num_tokens), time above 90%, prealloc queue.
Across ranks: "hot>90&cold<50" is the share of samples where one rank is above
90% active while another is below 50%.
"""
import collections
import datetime
import glob
import json
import statistics as st
import sys

R = sys.argv[1]
a = json.load(open(f"{R}/analysis/summary.json"))["points"]["80"]["phases"]["profiling"]
rs, rt, s0, e0 = a["rank_stages_ms"], a["rank_totals"], a["start_ns"], a["end_ns"]
hms = lambda ns: datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc).strftime("%H:%M:%S")
t0, t1 = hms(s0), hms(e0)
labels = sorted(rt["decode"])
ranks = [int(lab.rsplit("dp", 1)[1]) for lab in labels]

blocked = collections.defaultdict(set)
for p in glob.glob(f"{R}/diagnostics/decode/*.jsonl"):
    for line in open(p):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event") == "admission" and e.get("reason") == "kv_budget" and s0 <= e.get("wall_ns", 0) <= e0:
            blocked[e["dp_rank"]].add(e["rid"])

rows = []
for line in open(f"{R}/sampling/engine.jsonl"):
    d = json.loads(line)
    if "series" not in d or d.get("endpoint") != "decode-0" or not t0 <= d["captured_at"][11:19] <= t1:
        continue
    per = collections.defaultdict(dict)
    for s in d["series"]:
        if "dp_rank" in s.get("labels", {}):
            per[int(s["labels"]["dp_rank"])][s["metric"]] = s["value"]
    if len(per) == len(ranks):
        rows.append(per)

act = lambda m: (m.get("sglang:kv_used_tokens") or 0) / (m.get("sglang:max_total_num_tokens") or 1)
print("window", t0, t1, "samples", len(rows))
for k, lab in zip(ranks, labels):
    aw, gen = rs[f"decode/{lab}/alloc_wait_ms"], rs[f"decode/{lab}/generation_ms"]
    A = [act(r[k]) for r in rows]
    q = [r[k].get("sglang:num_decode_prealloc_queue_reqs") or 0 for r in rows]
    print(f"dp{k}: req {rt['decode'][lab]:5d} input {rt['decode_input_tokens'][lab] / 1e6:6.1f}M "
          f"blocked {len(blocked[k]):4d} alloc wait {aw['mean'] / 1000:5.1f}/{aw['p90'] / 1000:5.1f}s "
          f"gen {gen['mean'] / 1000:5.1f}s active {st.mean(A) * 100:5.1f}% "
          f">90% {sum(x > 0.9 for x in A) / len(A) * 100:5.1f}% queue {st.mean(q):5.2f}")
mx = [max(act(r[k]) for k in ranks) for r in rows]
mn = [min(act(r[k]) for k in ranks) for r in rows]
print("mean active", round(st.mean(act(r[k]) for r in rows for k in ranks) * 100, 1),
      "% | hot>90&cold<50", round(sum(x > 0.9 and y < 0.5 for x, y in zip(mx, mn)) / len(rows) * 100, 1),
      "% | all>85%", round(sum(y > 0.85 for y in mn) / len(rows) * 100, 1),
      "% | spread", round(st.mean(x - y for x, y in zip(mx, mn)) * 100, 1), "pp")
