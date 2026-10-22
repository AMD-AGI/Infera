#!/usr/bin/env python3
"""ms/step binned by #running-req from SGLang 'Decode batch' lines of ONE rank (DCP decode TP0, or one DP rank).
step_ms = (t_line - t_prev_line) / interval for consecutive lines with dt < cap; the line's own #running-req and
#token label the window ending at that line. Usage: step_by_running.yihou.py LOG [--rank 'TP0 EP0'] [--interval 40]"""
import argparse, re, datetime as dt, statistics as st, collections
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--rank", default="TP0 EP0")
ap.add_argument("--interval", type=int, default=40); ap.add_argument("--cap", type=float, default=10.0)
a = ap.parse_args()
pat = re.compile(r"^(\S+)Z .*Decode batch, #running-req: (\d+), #token: (\d+).*?accept len: ([\d.]+)")
rows = []
for line in open(a.log, errors="ignore"):
    if "Decode batch" not in line or a.rank not in line: continue
    m = pat.match(line)
    if m: rows.append((dt.datetime.fromisoformat(m.group(1)[:26]), int(m.group(2)), int(m.group(3)), float(m.group(4))))
bins = collections.defaultdict(list)
edges = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 64)]
for p, c in zip(rows, rows[1:]):
    d = (c[0] - p[0]).total_seconds()
    if d <= 0 or d > a.cap: continue
    for lo, hi in edges:
        if lo <= c[1] <= hi: bins[(lo, hi)].append((1000 * d / a.interval, c[2], c[3], c[1])); break
print(f"{'running':>8} {'n':>5} {'ms/step med':>11} {'mean':>6} {'ctx tok/req':>12} {'acc':>5}")
for lo, hi in edges:
    v = bins.get((lo, hi), [])
    if not v: continue
    ms = [x[0] for x in v]
    print(f"{lo:>3}-{hi:<4} {len(v):>5} {st.median(ms):>11.1f} {st.mean(ms):>6.1f} {st.mean(x[1]/x[3] for x in v):>12.0f} {st.mean(x[2] for x in v):>5.2f}")
