#!/usr/bin/env python3
"""Decode prefix hits as the router saw them vs what D reused (P8D4 notes, section 8).

Usage: router_decode_hits.py RUN_DIR [CREDIT]
Needs RUN/analysis/{summary,decode-local-prefix}.json (analyze.py, analyze_decode_prefix.py),
RUN/server-logs/router.log with routing_experiments at info, and RUN/sampling/engine.jsonl.
CREDIT is INFERA_R2_DECODE_HIT_CREDIT of the run (default 0.5).
"""
import collections
import datetime
import json
import re
import statistics as st
import sys

R = sys.argv[1]
credit = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
a = json.load(open(f"{R}/analysis/summary.json"))["points"]["80"]["phases"]["profiling"]
hms = lambda ns: datetime.datetime.fromtimestamp(ns / 1e9, datetime.timezone.utc).strftime("%H:%M:%S")
t0, t1 = hms(a["start_ns"]), hms(a["end_ns"])

ansi = re.compile(r"\x1b\[[0-9;]*m")
pick = re.compile(r"pick policy=\"kv-aware\" decision_id=(\d+) role=Decode .*?picked=\S+#dp(\d+) "
                  r"cache_hits=(\d+) request_blocks=(\d+)")
cand = re.compile(r"routing experiment candidate decision_id=(\d+) role=Decode .*?target=\S+#dp(\d+) "
                  r".*?demand_cost=Some\(([-\d.]+)\)")
block_size = int(re.search(r'"page_size": *(\d+)', open(f"{R}/launch/server-info/decode-0.json").read())[1])
picks, cands = {}, collections.defaultdict(dict)
for line in open(f"{R}/server-logs/router.log", errors="replace"):
    line = ansi.sub("", line)
    # docker logs --timestamps prefix, then the router's own timestamp
    ts = line.split()[1][11:19] if line[:4].isdigit() and len(line.split()) > 1 else ""
    if not t0 <= ts <= t1:
        continue
    if m := pick.search(line):
        picks[int(m[1])] = (int(m[2]), int(m[3]), int(m[4]))
    elif m := cand.search(line):
        cands[int(m[1])][int(m[2])] = float(m[3])

n = len(picks)
hit = [d for d, p in picks.items() if p[1] > 0]
hb, rb = sum(p[1] for p in picks.values()), sum(p[2] for p in picks.values())
decided = 0
for d in hit:
    dp, h, _ = picks[d]
    c = cands.get(d, {})
    if dp in c and len(c) > 1 and min(v for k, v in c.items() if k != dp) < c[dp] + credit * h * block_size:
        decided += 1
frac = [picks[d][1] / picks[d][2] for d in hit if picks[d][2]]
print(f"window {t0}-{t1}: D decisions {n}")
print(f"  router: picked rank had cached blocks {len(hit)} ({100 * len(hit) / n:.1f}%), "
      f"cached / request blocks {100 * hb / rb:.1f}%; hit fraction among those mean {100 * st.mean(frac):.1f}%")
print(f"  credit could have decided at most {decided} picks ({100 * decided / n:.1f}% of decisions): "
      f"another rank was cheaper before the picked rank's credit")

p = json.load(open(f"{R}/analysis/decode-local-prefix.json"))
print(f"  D reused: {p['requests_with_reuse']} / {p['completed_requests']} requests "
      f"({100 * p['requests_with_reuse'] / p['completed_requests']:.1f}%), prefix tokens "
      f"{100 * p['prefix_tokens_last_allocation'] / p['input_tokens_last_allocation']:.1f}% of input")

ev, used, cap = [], [], 0
for line in open(f"{R}/sampling/engine.jsonl"):
    d = json.loads(line)
    if "series" not in d or d.get("endpoint") != "decode-0" or not t0 <= d["captured_at"][11:19] <= t1:
        continue
    per = collections.defaultdict(dict)
    for s in d["series"]:
        if "dp_rank" in s.get("labels", {}):
            per[int(s["labels"]["dp_rank"])][s["metric"]] = s["value"]
    cap = sum(m.get("sglang:max_total_num_tokens") or 0 for m in per.values())
    ev.append(sum(m.get("sglang:kv_evictable_tokens") or 0 for m in per.values()))
    used.append(sum(m.get("sglang:kv_used_tokens") or 0 for m in per.values()))
q = lambda v, x: sorted(v)[int(x * (len(v) - 1))]
print(f"  D cache headroom: capacity {cap / 1e6:.2f}M tokens, active {100 * st.mean(used) / cap:.1f}%, evictable "
      f"mean {st.mean(ev) / 1e6:.2f}M p50 {q(ev, .5) / 1e6:.2f}M p10 {q(ev, .1) / 1e6:.2f}M")
