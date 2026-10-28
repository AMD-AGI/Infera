#!/usr/bin/env python3
"""Decode DP-rank balance from a decode log ("Decode batch" lines): per rank token usage, running, prealloc and transfer
queues over the scored window, the spread across ranks, and how often a rank is short of KV (prealloc queue > 0) while
another rank still has room.
  python3 decode_dp_balance.py DECODE_LOG START_HH:MM:SS END_HH:MM:SS [--room 0.80]"""

import argparse
import collections
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("start")
ap.add_argument("end")
ap.add_argument("--room", type=float, default=0.80, help="token usage below which a rank counts as having room")
a = ap.parse_args()

pat = re.compile(r"\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d) DP(\d) TP\d+\] Decode batch, #running-req: (\d+), #token: (\d+), "
                 r"token usage: ([\d.]+).*?pre-allocated usage: ([\d.]+), #prealloc-req: (\d+), #transfer-req: (\d+)")
last = {}
bins = collections.OrderedDict()
for line in open(a.log, errors="replace"):
    m = pat.search(line)
    if not m:
        continue
    t = m.group(1)
    if not (a.start <= t <= a.end) if a.start <= a.end else not (t >= a.start or t <= a.end):
        continue
    r = int(m.group(2))
    last[r] = (int(m.group(3)), int(m.group(4)), float(m.group(5)), float(m.group(6)), int(m.group(7)), int(m.group(8)))
    if len(last) == 4:
        bins[t] = dict(last)

rows = list(bins.values())
print(f"{len(rows)} one-second samples with all 4 ranks")
for name, k in (("running", 0), ("token usage", 2), ("prealloc usage", 3), ("prealloc-req", 4), ("transfer-req", 5)):
    per = [[row[r][k] for row in rows] for r in range(4)]
    print(f"  {name:15s} " + "  ".join(f"DP{r} p50 {st.median(per[r]):6.2f} mean {st.mean(per[r]):6.2f}" for r in range(4)))
spread_run = [max(row[r][0] for r in range(4)) / max(1e-9, st.mean(row[r][0] for r in range(4))) for row in rows]
spread_use = [max(row[r][2] + row[r][3] for r in range(4)) - min(row[r][2] + row[r][3] for r in range(4)) for row in rows]
print(f"  busiest / mean running: p50 {st.median(spread_run):.2f} p90 {sorted(spread_run)[int(.9 * len(rows))]:.2f}")
print(f"  max - min (usage + prealloc usage): p50 {st.median(spread_use):.2f} p90 {sorted(spread_use)[int(.9 * len(rows))]:.2f}")
any_wait = [row for row in rows if any(row[r][4] > 0 for r in range(4))]
stranded = [row for row in any_wait if any(row[r][2] + row[r][3] < a.room for r in range(4))]
print(f"  samples with a prealloc queue on some rank: {len(any_wait) / len(rows):.0%}; "
      f"of those, another rank below {a.room:.0%} usage: {len(stranded) / max(1, len(any_wait)):.0%}")
tot = [sum(row[r][0] for r in range(4)) for row in rows]
pre = [sum(row[r][4] for r in range(4)) for row in rows]
tr = [sum(row[r][5] for r in range(4)) for row in rows]
print(f"  total running p50 {st.median(tot)}, prealloc-req mean {st.mean(pre):.1f}, transfer-req mean {st.mean(tr):.1f}")
