#!/usr/bin/env python3
"""Per-DP-rank load balance over time, from rank_samples.jsonl.

prefill_batches is CUMULATIVE, so the per-tick delta is the real instantaneous
load. Reports the coefficient of variation (CV = std/mean) per tick: 0 is
perfect balance, and >0.5 means one rank is doing much more than another.
"""
import json, sys, statistics as st

p = sys.argv[1] if len(sys.argv) > 1 else "/shared_nfs/yihou_agentx_caseA/results/rank_samples.jsonl"
rows = [json.loads(l) for l in open(p) if l.strip()]
if len(rows) < 2:
    print("not enough samples yet"); sys.exit()

ranks = sorted({r for row in rows for r in row["prefill_batches"]},
               key=lambda x: int(x[2:]))
print(f"samples={len(rows)}  window={rows[0]['t']} -> {rows[-1]['t']}  ranks={len(ranks)}\n")

# cumulative totals over the whole window
first, last = rows[0]["prefill_batches"], rows[-1]["prefill_batches"]
tot = {r: last.get(r, 0) - first.get(r, 0) for r in ranks}
s = sum(tot.values())
print("cumulative prefill batches over the window:")
for r in ranks:
    share = 100 * tot[r] / s if s else 0
    bar = "#" * int(share * 1.2)
    print(f"  {r:5} {tot[r]:7,}  {share:5.1f} %  {bar}")
if s:
    vals = [tot[r] for r in ranks]
    m = st.mean(vals)
    cv = st.pstdev(vals) / m if m else 0
    print(f"\n  total={s:,}  mean={m:,.0f}  min={min(vals):,}  max={max(vals):,}"
          f"  max/min={max(vals)/max(min(vals),1):.2f}x  CV={cv:.3f}")

# per-tick CV trend
print("\nper-tick delta CV (0 = perfectly balanced):")
cvs = []
for a, b in zip(rows, rows[1:]):
    d = [b["prefill_batches"].get(r, 0) - a["prefill_batches"].get(r, 0) for r in ranks]
    tot_d = sum(d)
    if tot_d <= 0:
        continue
    m = st.mean(d)
    cvs.append((b["t"], tot_d, st.pstdev(d) / m if m else 0))
if cvs:
    for t, n, c in cvs[-12:]:
        print(f"  {t}  batches={n:5}  CV={c:.3f}")
    allc = [c for _, _, c in cvs]
    print(f"\n  active ticks={len(cvs)}  mean CV={st.mean(allc):.3f}  median={st.median(allc):.3f}")
else:
    print("  (no active ticks yet)")
