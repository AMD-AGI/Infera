#!/usr/bin/env python3
"""Per-request prefill computation vs the ideal, for a P4D4 run with USAGE_LOG=1 (the router logs every prefill's
prompt / cached tokens). Each AIPerf record (profile_export.jsonl) is matched to a router line with the same prompt
token count that was logged after the request started. For a request whose conversation had an earlier turn in the
run (warmup included), the ideal computation is ISL - ISL(previous turn) (append-only history, the previous prompt
fully cached); without an earlier turn it is a cold start (ideal: everything, unless another conversation shares the
prefix). Prints where the computed tokens go (cold / ideal new tokens / excess over the ideal) and the excess
distribution, for the measurement phase.
  prefill_usage.py RUN_DIR [N_GPUS=8]
"""
import bisect
import collections
import gzip
import json
import re
import sys
from datetime import datetime, timezone

run = sys.argv[1].rstrip("/")
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"^(\S+Z) .*prefill_usage id=(\S*) prompt=(-?\d+) cached=(-?\d+) leg_ms=(\d+)")
logs = collections.defaultdict(list)  # prompt -> [(t_end, cached, leg_ms)]
for name in ("vpd-router.stream.log.gz", "vpd-router.log.gz", "vpd-router.stream.log", "vpd-router.log"):
    try:
        op = gzip.open if name.endswith(".gz") else open
        with op(f"{run}/{name}", "rt", errors="replace") as f:
            for line in f:
                m = pat.match(ansi.sub("", line))
                if m:
                    t = datetime.strptime(m.group(1)[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc)
                    logs[int(m.group(3))].append((t.timestamp(), int(m.group(4)), int(m.group(5))))
        break
    except FileNotFoundError:
        continue
for v in logs.values():
    v.sort()
n_log = sum(len(v) for v in logs.values())
recs = []
for line in open(f"{run}/agentx-c104/aiperf_artifacts/profile_export.jsonl"):
    r = json.loads(line)
    md, m = r.get("metadata", {}), r.get("metrics") or {}
    isl = (m.get("usage_prompt_tokens") or {}).get("value")
    if isl is None:
        continue
    recs.append(dict(conv=md.get("conversation_id"), turn=md.get("turn_index"), phase=md.get("benchmark_phase"),
                     depth=md.get("agent_depth"), t0=md["request_start_ns"] / 1e9, isl=int(isl)))
used = set()
matched = 0
for r in sorted(recs, key=lambda x: x["t0"]):
    cand = logs.get(r["isl"], [])
    i = bisect.bisect_left(cand, (r["t0"],))
    while i < len(cand) and (r["isl"], i) in used:
        i += 1
    if i < len(cand) and cand[i][0] - r["t0"] < 600:
        used.add((r["isl"], i))
        r["cached"], r["leg_ms"] = cand[i][1], cand[i][2]
        matched += 1
print(f"{run}: {n_log} router prefill lines, {len(recs)} AIPerf records, {matched} matched")
by_conv = collections.defaultdict(list)
for r in recs:
    by_conv[r["conv"]].append(r)
for v in by_conv.values():
    v.sort(key=lambda x: (x["turn"], x["t0"]))
    for a, b in zip([None] + v[:-1], v):
        b["prev_isl"] = a["isl"] if a is not None and a["isl"] <= b["isl"] else None
cat = collections.Counter()
ncat = collections.Counter()
excess = []
for r in recs:
    if r["phase"] != "profiling" or "cached" not in r or r["cached"] < 0:
        continue
    comp = r["isl"] - r["cached"]
    if r["prev_isl"] is None:
        cat["cold: computed"] += comp
        ncat["cold"] += 1
        continue
    ideal = r["isl"] - r["prev_isl"]
    cat["warm: ideal new tokens"] += min(comp, ideal)
    cat["warm: excess over ideal"] += max(comp - ideal, 0)
    ncat["warm"] += 1
    excess.append((comp - ideal, r))
tot = sum(cat.values())
isl_tot = sum(r["isl"] for r in recs if r["phase"] == "profiling" and "cached" in r)
print(f"profiled, matched: {ncat['cold']} cold + {ncat['warm']} warm requests, ISL {isl_tot:,}, computed {tot:,} "
      f"({tot / max(isl_tot, 1):.3%} of ISL)")
for k, v in cat.most_common():
    print(f"  {k:28s} {v:14,} tokens  {v / max(tot, 1):6.1%} of computed  {v / max(isl_tot, 1):6.2%} of ISL")
if excess:
    xs = sorted(e for e, _ in excess)
    q = lambda p: xs[min(int(p * len(xs)), len(xs) - 1)]
    print(f"warm excess per request: p10 {q(0.1)} p50 {q(0.5)} p75 {q(0.75)} p90 {q(0.9)} p99 {q(0.99)} max {xs[-1]}")
    bins = [(-10**9, 0, "<= 0"), (0, 64, "1-64"), (64, 512, "65-512"), (512, 4096, "513-4K"), (4096, 32768, "4K-32K"),
            (32768, 10**9, "> 32K")]
    for lo, hi, lab in bins:
        sel = [e for e, _ in excess if lo < e <= hi]
        print(f"  excess {lab:>7s}: {len(sel):6d} requests, {sum(max(e, 0) for e in sel):14,} excess tokens")
    big = sorted(excess, key=lambda x: -x[0])[:8]
    for e, r in big:
        print(f"    excess {e:7d}: isl {r['isl']:7d} prev {r['prev_isl']:7d} cached {r['cached']:7d} depth {r['depth']} "
              f"turn {r['turn']} conv {str(r['conv'])[:40]}")
