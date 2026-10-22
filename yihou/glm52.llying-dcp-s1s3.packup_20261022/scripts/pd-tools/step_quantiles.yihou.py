#!/usr/bin/env python3
"""p50/p90 ms/step binned by decode #running-req, one rank. Same step definition as pdrun/step_by_running.yihou.py:
step_ms = (t_line - t_prev_line) / interval for consecutive 'Decode batch' lines with 0 < dt <= cap.
Usage: step_quantiles.yihou.py LOG [--rank 'TP0 EP0'] [--interval 40]"""
import argparse, re, datetime as dt, collections
ap = argparse.ArgumentParser(); ap.add_argument("log"); ap.add_argument("--rank", default="TP0 EP0")
ap.add_argument("--interval", type=int, default=40); ap.add_argument("--cap", type=float, default=10.0)
a = ap.parse_args()
pat = re.compile(r"^(\S+)Z .*Decode batch, #running-req: (\d+), #token: (\d+)")
rows = []
for line in open(a.log, errors="ignore"):
    if "Decode batch" not in line or a.rank not in line: continue
    m = pat.match(line)
    if m: rows.append((dt.datetime.fromisoformat(m.group(1)[:26]), int(m.group(2))))
edges = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 64)]
bins = collections.defaultdict(list); allv = []
for p, c in zip(rows, rows[1:]):
    d = (c[0] - p[0]).total_seconds()
    if d <= 0 or d > a.cap: continue
    v = 1000 * d / a.interval; allv.append(v)
    for lo, hi in edges:
        if lo <= c[1] <= hi: bins[(lo, hi)].append(v); break
def q(v, f):
    s = sorted(v); return s[min(len(s) - 1, int(f * (len(s) - 1) + 0.5))]
print(f"{'running':>8} {'n':>5} {'p50':>6} {'p90':>6}")
for lo, hi in edges:
    v = bins.get((lo, hi), [])
    if v: print(f"{lo:>3}-{hi:<4} {len(v):>5} {q(v,.5):>6.1f} {q(v,.9):>6.1f}")
print(f"{'all':>8} {len(allv):>5} {q(allv,.5):>6.1f} {q(allv,.9):>6.1f}")
