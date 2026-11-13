#!/usr/bin/env python3
"""Per-rank dependence of context-driven kernels on the rank's OWN running requests (not the padded graph batch):
least-squares us/call = a + b * own_bs for chosen classes, over all ranks and steps of the given ana2 JSONs."""
import json, sys, statistics as st, collections
cls = sys.argv[1].split(","); files = sys.argv[2:]
pts = collections.defaultdict(list)
for f in files:
    r = json.load(open(f))
    for s in r["steps"]:
        for k in range(r["ranks"]):
            for c in cls:
                n = s["cls_n"][k].get(c, 0)
                if n:
                    pts[c].append((s["bs"][k], s["gbs"], s["cls_t"][k][c] / n))
for c in cls:
    P = pts[c]
    if not P:
        continue
    xs = [p[0] for p in P]; ys = [p[2] for p in P]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys)); st_ = sum((y - my) ** 2 for y in ys)
    print(f"{c:12s} n={len(P):5d} us/call = {a:6.2f} + {b:5.2f} * own_bs   R^2 {1 - ss / st_:.3f}   own_bs range {min(xs)}..{max(xs)}")
    byb = collections.defaultdict(list)
    for x, g, y in P:
        byb[x].append(y)
    print("             own_bs: " + " ".join(f"{x}:{st.mean(v):.0f}" for x, v in sorted(byb.items())))
