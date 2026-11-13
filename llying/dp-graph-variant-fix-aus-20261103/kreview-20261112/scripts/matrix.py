#!/usr/bin/env python3
"""class x graph-batch matrix (ms/step mean over ranks; optional us/call) pooled over the given ana2 JSONs."""
import collections, json, statistics as st, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from ana2 import ORDER
mode = sys.argv[1]  # ms | us | n
files = sys.argv[2:]
by = collections.defaultdict(list); R = None
for f in files:
    r = json.load(open(f)); R = r["ranks"]
    for s in r["steps"]:
        by[s["gbs"]].append(s)
gs = [g for g in sorted(by) if len(by[g]) >= 3]
print(f"{'class':12s} " + " ".join(f"{g:7d}" for g in gs))
print(f"{'steps':12s} " + " ".join(f"{len(by[g]):7d}" for g in gs))
print(f"{'mean bs':12s} " + " ".join(f"{st.mean(st.mean(s['bs']) for s in by[g]):7.1f}" for g in gs))
tot = {g: 0.0 for g in gs}
for c in ORDER:
    row = []
    for g in gs:
        S = by[g]
        t = st.mean(st.mean(s["cls_t"][k].get(c, 0.0) for k in range(len(s["cls_t"]))) for s in S) / 1e3
        n = st.mean(st.mean(s["cls_n"][k].get(c, 0) for k in range(len(s["cls_n"]))) for s in S)
        tot[g] += t
        row.append(t if mode == "ms" else (1e3 * t / n if n else 0.0) if mode == "us" else n)
    if any(row):
        print(f"{c:12s} " + " ".join(f"{x:7.2f}" if mode != "n" else f"{x:7.0f}" for x in row))
print(f"{'verify':12s} " + " ".join(f"{tot[g]:7.2f}" for g in gs))
print(f"{'kernels':12s} " + " ".join(f"{st.mean(st.mean(s['nk']) for s in by[g]):7.0f}" for g in gs))
w = []
for g in gs:
    x = 0.0
    for s in by[g]:
        sy = s["sync"]; kind = "mega" if "mega_min" in sy else "ag"
        x += (sy.get(kind + "_mean", 0) - sy.get(kind + "_min", 0) + sy.get("rs_mean", 0) - sy.get("rs_min", 0)) / len(by[g]) / 1e3
    w.append(x)
print(f"{'(waiting)':12s} " + " ".join(f"{x:7.2f}" for x in w))
print(f"{'loc max-mean':12s} " + " ".join(f"{st.mean((s['sync'].get('local_max', 0) - s['sync'].get('local_mean', 0)) for s in by[g]) / 1e3:7.2f}" for g in gs))
