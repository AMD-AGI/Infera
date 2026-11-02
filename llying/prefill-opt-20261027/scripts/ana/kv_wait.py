"""Decode KV pool pressure by DP rank: token usage, requests waiting for KV preallocation, and whether the waiting rank is
full while another has room.   python3 kv_wait.py DECODE_LOG T0 T1"""
import re, sys, statistics as st
from datetime import datetime
log, T0, T1 = sys.argv[1:4]
R = re.compile(r"^(\S+?)\d{3}Z \[\S+ \S+ DP(\d) TP\d\] Decode batch, #running-req: (\d+), #token: (\d+), token usage: ([\d.]+).*pre-allocated usage: ([\d.]+), #prealloc-req: (\d+), #transfer-req: (\d+)")
last = {}; rows = []
for line in open(log, errors="replace"):
    m = R.match(line)
    if not m or not (T0 <= m.group(1)[:19] < T1): continue
    t = datetime.fromisoformat(m.group(1)).timestamp(); dp = int(m.group(2))
    last[dp] = (t, int(m.group(3)), float(m.group(5)), float(m.group(6)), int(m.group(7)), int(m.group(8)))
    cur = {k: v for k, v in last.items() if t - v[0] < 3}
    if len(cur) == 4: rows.append((t, dict(cur)))
n = len(rows); print("samples", n)
us = [[r[1][d][2] for r in rows] for d in range(4)]
print("token usage per rank mean", [round(st.mean(u), 3) for u in us], " p90", [round(sorted(u)[int(.9 * n)], 3) for u in us], " share >=0.97", [round(sum(x >= .97 for x in u) / n, 2) for u in us])
pre = [[r[1][d][4] for r in rows] for d in range(4)]
print("prealloc-req per rank mean", [round(st.mean(p), 2) for p in pre], " total mean %.1f" % st.mean(sum(r[1][d][4] for d in range(4)) for r in rows), " share of samples with any waiting %.2f" % (sum(any(r[1][d][4] > 0 for d in range(4)) for r in rows) / n))
w = [(r[1][d][2], min(r[1][e][2] for e in range(4) if e != d), r[1][d][3]) for r in rows for d in range(4) if r[1][d][4] > 0]
if w:
    print("when a rank has waiting requests: its usage mean %.3f; the least-used OTHER rank's usage mean %.3f; share where another rank has >=10%% (0.10) more room: %.2f" % (st.mean(x[0] for x in w), st.mean(x[1] for x in w), sum(x[0] - x[1] >= .10 for x in w) / len(w)))
    print("   ... >= 0.20 more room: %.2f" % (sum(x[0] - x[1] >= .20 for x in w) / len(w)))
print("pre-allocated usage (reserved for incoming) mean per rank", [round(st.mean(r[1][d][3] for r in rows), 3) for d in range(4)])
