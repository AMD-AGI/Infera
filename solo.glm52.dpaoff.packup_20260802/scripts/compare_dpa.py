#!/usr/bin/env python3
"""Compare the DPA-off solo run against the DPA-on solo baseline.

Both runs use the SAME workload file, the SAME seed, and the SAME driver patch,
at concurrency exactly 1. The only deployment difference is the prefill leg's
attention sharding (DP8 -> pure TP8, with ep_size held at 8 so MoE is unchanged).

Bucketing by input length is not cosmetic: TTFT is dominated by prefill work,
which scales with input size. An unbucketed ratio silently mixes "the service
got faster" with "this run happened to draw smaller prompts". The two runs'
input ladders agree closely here, so both views are shown -- but the bucketed
one is the load-bearing table.
"""
import json, gzip, math, sys

def load(path, phase="sustain"):
    op = gzip.open if path.endswith(".gz") else open
    F = {k: [] for k in ("ttft", "e2e", "tpot", "inp", "gen", "acc", "chr")}
    rows = 0
    for line in op(path, "rt"):
        r = json.loads(line)
        if r.get("phase") != phase:
            continue
        rows += 1
        F["ttft"] += r.get("new_ttfts") or []
        F["e2e"]  += r.get("new_e2es") or []
        F["tpot"] += r.get("new_tpots") or []
        F["inp"]  += r.get("new_prompt_lengths") or []
        F["gen"]  += r.get("new_generation_lengths") or []
        F["acc"]  += r.get("new_acceptance_lengths") or []
        F["chr"]  += r.get("new_cache_hit_rates") or []
    F["_rows"] = rows
    return F

def P(a, p):
    if not a: return float("nan")
    a = sorted(a); k = (len(a)-1)*p/100.0
    lo, hi = math.floor(k), math.ceil(k)
    return a[lo] if lo == hi else a[lo] + (a[hi]-a[lo])*(k-lo)

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_KIT = os.path.dirname(_HERE)
# Defaults point INSIDE this kit, so the comparison reproduces with no sibling
# checkout and no arguments:  python3 scripts/compare_dpa.py
_DEF_ON = os.path.join(_KIT, "results", "baseline_dpaon_metrics.jsonl.gz")
_DEF_OFF = os.path.join(_KIT, "results", "metrics.jsonl.gz")

ON  = load(sys.argv[1] if len(sys.argv) > 1 else _DEF_ON)   # DPA-on baseline
OFF = load(sys.argv[2] if len(sys.argv) > 2 else _DEF_OFF)  # DPA-off

print("# prefill DPA: OFF vs ON — solo (concurrency 1), identical workload+seed\n")
print(f"samples: DPA-on n={len(ON['ttft'])}   DPA-off n={len(OFF['ttft'])}\n")

# --- sanity: the request shapes must match, else nothing below is comparable --
print("## Shape parity check (must match, or the comparison is void)\n")
print("| | DPA-on | DPA-off | delta |")
print("|---|---|---|---|")
for lbl, key, f in (("input p50", "inp", "{:,.0f}"), ("input mean", "inp", "{:,.0f}"),
                    ("output p50", "gen", "{:,.0f}"), ("output mean", "gen", "{:,.0f}")):
    a = P(ON[key],50) if "p50" in lbl else sum(ON[key])/len(ON[key])
    b = P(OFF[key],50) if "p50" in lbl else sum(OFF[key])/len(OFF[key])
    print(f"| {lbl} | {f.format(a)} | {f.format(b)} | {(b-a)/a*100:+.1f}% |")

def ladder(name, key, scale=1000.0, fmt="{:,.0f}", pcts=(50,90,99)):
    print(f"\n## {name}\n")
    print("| stat | DPA-on | DPA-off | change |")
    print("|---|---|---|---|")
    a_all = [x for x in ON[key] if x > 0]
    b_all = [x for x in OFF[key] if x > 0]
    rows = [("min", min(a_all)*scale, min(b_all)*scale)]
    for p in pcts:
        rows.append((f"**p{p}**", P(a_all,p)*scale, P(b_all,p)*scale))
    rows.append(("mean", sum(a_all)/len(a_all)*scale, sum(b_all)/len(b_all)*scale))
    for lbl, a, b in rows:
        star = "**" if lbl.startswith("**") else ""
        print(f"| {lbl} | {star}{fmt.format(a)}{star} | {star}{fmt.format(b)}{star} | "
              f"{star}{(b-a)/a*100:+.1f}%{star} |")

ladder("TTFT (ms) — the metric prefill DPA should move", "ttft")
ladder("TPOT (ms) — decode leg untouched, expect ~flat", "tpot", 1000.0, "{:,.2f}")
ladder("E2E (ms)", "e2e")
ladder("MTP acceptance", "acc", 1.0, "{:.3f}")

# --- the load-bearing table: TTFT bucketed by input length -------------------
BUCKETS = [(0,50_000,"0-50K"), (50_000,80_000,"50-80K"), (80_000,120_000,"80-120K"),
           (120_000,160_000,"120-160K"), (160_000,10**9,"160-300K")]
print("\n## TTFT by input-length bucket (the load-bearing comparison)\n")
print("| input | n(on) | n(off) | mean TTFT on | mean TTFT off | change | speedup |")
print("|---|---|---|---|---|---|---|")
for lo, hi, lbl in BUCKETS:
    a = [t for t, i in zip(ON["ttft"], ON["inp"]) if lo <= i < hi]
    b = [t for t, i in zip(OFF["ttft"], OFF["inp"]) if lo <= i < hi]
    if not a or not b:
        print(f"| {lbl} | {len(a)} | {len(b)} | — | — | — | — |")
        continue
    ma, mb = sum(a)/len(a)*1000, sum(b)/len(b)*1000
    print(f"| {lbl} | {len(a)} | {len(b)} | {ma:,.0f} ms | {mb:,.0f} ms | "
          f"{(mb-ma)/ma*100:+.1f}% | **{ma/mb:.2f}x** |")

# --- TPOT bucketed by output length: is the decode leg really unchanged? -----
OB = [(0,100,"0-100"), (100,500,"100-500"), (500,2000,"500-2K"), (2000,20000,"2-6K+")]
print("\n## TPOT by output-length bucket (decode leg was NOT restarted)\n")
print("| output tok | n(on) | n(off) | mean TPOT on | mean TPOT off | change |")
print("|---|---|---|---|---|---|")
for lo, hi, lbl in OB:
    a = [t*1000 for t, g in zip(ON["tpot"], ON["gen"]) if lo <= g < hi and t > 0]
    b = [t*1000 for t, g in zip(OFF["tpot"], OFF["gen"]) if lo <= g < hi and t > 0]
    if not a or not b:
        print(f"| {lbl} | {len(a)} | {len(b)} | — | — | — |"); continue
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    print(f"| {lbl} | {len(a)} | {len(b)} | {ma:.2f} ms | {mb:.2f} ms | {(mb-ma)/ma*100:+.1f}% |")

print("\n## Cache hit rate (construction target 89.0%)\n")
for n, F in (("DPA-on", ON), ("DPA-off", OFF)):
    c = F["chr"]
    print(f"  {n}: mean {sum(c)/len(c)*100:.2f}%  p50 {P(c,50)*100:.2f}%")

# --- paired comparison: the cleanest evidence -------------------------------
# Both runs share seed 1337, so many (input, output) shapes recur in both. This
# compares each request against ITSELF, removing the "did this run happen to
# draw easier prompts" confound entirely.
def keyed(F):
    d = {}
    for t, i, g in zip(F["ttft"], F["inp"], F["gen"]):
        d.setdefault((i, g), []).append(t * 1000)
    return d

kon, koff = keyed(ON), keyed(OFF)
common = sorted(set(kon) & set(koff))
if common:
    import statistics
    rs = []
    for k in common:
        a = sum(kon[k]) / len(kon[k])
        b = sum(koff[k]) / len(koff[k])
        rs.append((a / b, k, a, b))
    faster = sum(1 for r, _, _, _ in rs if r > 1)
    rs.sort()
    print("\n## Paired on identical (input, output) shapes — the cleanest evidence\n")
    print(f"  paired shapes present in both runs : {len(common)}")
    print(f"  DPA-off faster on                  : {faster}/{len(common)}")
    print(f"  MEDIAN PAIRED SPEEDUP              : {statistics.median([r for r,_,_,_ in rs]):.2f}x")
    print("\n  worst 3 for DPA-off (<1.00x = regression):")
    for r, k, a, b in rs[:3]:
        print(f"    in={k[0]:>7,} out={k[1]:>6,}  on={a:8,.0f}ms  off={b:8,.0f}ms  {r:.2f}x")
    print("  best 3:")
    for r, k, a, b in rs[-3:]:
        print(f"    in={k[0]:>7,} out={k[1]:>6,}  on={a:8,.0f}ms  off={b:8,.0f}ms  {r:.2f}x")

# --- the other half of the trade: aggregate KV capacity ---------------------
# Constants read from the two legs' boot logs (see environment.md). Under DPA
# each of the 8 DP ranks owns a DISTINCT KV shard; under pure TP there is one
# scheduler and one pool, replicated across TP ranks.
ON_PER, OFF_PER, NRANK = 2_829_952, 3_263_680, 8
print("\n## The other half of the trade: aggregate KV capacity\n")
print("| | per rank | distinct shards | aggregate addressable KV |")
print("|---|---|---|---|")
print(f"| DPA-on | {ON_PER:,} | {NRANK} | **{NRANK*ON_PER:,}** |")
print(f"| DPA-off | {OFF_PER:,} | 1 (replicated) | **{OFF_PER:,}** |")
print(f"\n  per-rank change   : {(OFF_PER/ON_PER-1)*100:+.1f}%  (attention weights shard, freeing memory)")
print(f"  AGGREGATE change  : {(OFF_PER/(NRANK*ON_PER)-1)*100:+.1f}%  <- this is what drives eviction")
print("\n  DP-attention spends latency to buy capacity. This run measures both"
      "\n  currencies at once: TTFT halves, aggregate KV falls ~86%.")
