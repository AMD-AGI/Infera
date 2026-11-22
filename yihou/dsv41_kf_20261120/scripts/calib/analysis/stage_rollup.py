#!/usr/bin/env python3
"""Roll up a vllm_table.py JSON over selected stages: bucket and kernel shares, mHC rank, per-stage token stats.
usage: stage_rollup.py TABLE_JSON --stages DECODE,SAMPLE_after_DECODE [--exclude-comm] [--top 25]"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from vllm_table import bucket_of  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("table_json"); p.add_argument("--stages", required=True); p.add_argument("--exclude-comm", action="store_true")
p.add_argument("--top", type=int, default=25)
a = p.parse_args()
d = json.load(open(a.table_json)); want = set(a.stages.split(","))
tot = defaultdict(float); calls = defaultdict(float); recs = 0; nrank = 0
for eng, ranks in d["engines"].items():
    for r in ranks:
        nrank += 1
        for st, S in r["stages"].items():
            if st not in want: continue
            if st in ("DECODE", "MIXED"): recs += S["records"]
            for k, c, v, t, m, b, g in S["kernels"]:
                if a.exclude_comm and b == "comm": continue
                tot[k] += t; calls[k] += c * S["records"]
g = sum(tot.values()) or 1
bk = defaultdict(float)
for k, v in tot.items(): bk[bucket_of(k)] += v
br = sorted(bk.items(), key=lambda x: -x[1])
print(f"stages={sorted(want)} ranks={nrank} step-records={recs} kernel-ms={g:.1f} exclude_comm={a.exclude_comm}")
print("| bucket | ms | share |\n|---|---|---|")
for b, v in br: print(f"| {b} | {v:.1f} | {100*v/g:.1f}% |")
ks = sorted(tot.items(), key=lambda x: -x[1])
print("\n| # | kernel | ms | share | calls/step | bucket |\n|---|---|---|---|---|---|")
for i, (k, v) in enumerate(ks[:a.top]):
    print(f"| {i+1} | `{k[:95]}` | {v:.1f} | {100*v/g:.2f}% | {calls[k]/max(recs,1):.1f} | {bucket_of(k)} |")
mh = [(i + 1, k, v) for i, (k, v) in enumerate(ks) if bucket_of(k) == "mhc"]
print(f"\nmHC: bucket rank {[b for b,_ in br].index('mhc')+1 if 'mhc' in bk else None}, share {100*bk.get('mhc',0)/g:.2f}%, "
      f"kernels {len(mh)}, first kernel rank {mh[0][0] if mh else None}")
for i, k, v in mh: print(f"  #{i} {k[:110]} {v:.1f} ms ({100*v/g:.2f}%) calls/step {calls[k]/max(recs,1):.1f}")
