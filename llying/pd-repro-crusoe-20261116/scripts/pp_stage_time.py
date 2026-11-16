#!/usr/bin/env python3
"""PP prefill per-stage time against the batch's cached context, the measures of
llying/dp-graph-variant-fix-aus-20261103/BOTTLENECK-PP8-20261109.zh-CN.md section 4b, from one prefill container log
(docker timestamps). The i-th "Prefill batch" line of PP k and of PP k+1 with the same (#new-token, #cached-token) is the
same batch; their timestamp difference is the single-stage time of PP k.
Usage: pp_stage_time.py PREFILL_LOG START_ISO END_ISO"""
import collections
import datetime as dt
import re
import statistics as st
import sys

log, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3]
pat = re.compile(r"^(\S+) \[[^\]]* PP(\d+)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+).*?#queue-req: (\d+)")
rows = collections.defaultdict(list)
for line in open(log, errors="replace"):
    m = pat.match(line)
    if not m:
        continue
    ts = m.group(1)[:26]
    if not (t0 <= ts[:19] <= t1):
        continue
    t = dt.datetime.fromisoformat(ts).timestamp()
    rows[int(m.group(2))].append((t, int(m.group(3)), int(m.group(4)), int(m.group(5)), int(m.group(6))))
pps = sorted(rows)
if not pps:
    sys.exit("no Prefill batch lines in the window")
p0 = rows[pps[0]]
n = len(p0)
gaps = [b[0] - a[0] for a, b in zip(p0, p0[1:])]
print(f"window {t0} .. {t1}; PP ranks {pps}; PP0 batches {n} ({n / max(1, p0[-1][0] - p0[0][0]):.2f}/s)")
print(f"per batch: new tokens mean {st.mean(r[2] for r in p0):,.0f}, requests {st.mean(r[1] for r in p0):.2f}, "
      f"cached mean {st.mean(r[3] for r in p0):,.0f} median {st.median(r[3] for r in p0):,.0f}; queue mean {st.mean(r[4] for r in p0):.1f}, "
      f"batches with a queue {100 * sum(r[4] > 0 for r in p0) / n:.0f}%")
print(f"PP0 batch interval: median {1000 * st.median(gaps):.0f} ms, p10 {1000 * sorted(gaps)[len(gaps) // 10]:.0f}, "
      f"p90 {1000 * sorted(gaps)[9 * len(gaps) // 10]:.0f}, mean {1000 * st.mean(gaps):.0f}")
print(f"uncached tok/s over the window: {sum(r[2] for r in p0) / (p0[-1][0] - p0[0][0]):,.0f}")
# match batch i of PP k to PP k+1 by order and identical (new, cached)
stage = collections.defaultdict(list)  # (k) -> [(dt, cached, new)]
for k, k1 in zip(pps, pps[1:]):
    a, b = rows[k], rows[k1]
    j = 0
    for t, s, new, cached, q in a:
        while j < len(b) and b[j][0] < t:
            j += 1
        for jj in range(j, min(j + 4, len(b))):
            if b[jj][2] == new and b[jj][3] == cached:
                stage[k].append((b[jj][0] - t, cached, new))
                j = jj + 1
                break
print("single-stage time, median ms (n): " + ", ".join(
    f"PP{k}->{k + 1} {1000 * st.median(d for d, _, _ in v):.0f} ({len(v)})" for k, v in sorted(stage.items())))
allv = [x for v in stage.values() for x in v]
bins = [(0, 20e3), (20e3, 80e3), (80e3, 160e3), (160e3, 250e3), (250e3, 400e3), (400e3, 700e3), (700e3, 1e9)]
print("single-stage time by cached context of the batch (all adjacent pairs; full 8K-token chunks / all batches):")
for lo, hi in bins:
    v = [d for d, c, nw in allv if lo <= c < hi]
    f = [d for d, c, nw in allv if lo <= c < hi and nw >= 8192]
    if v:
        print(f"  cached {lo / 1e3:>4.0f}K-{hi / 1e3 if hi < 1e8 else float('inf'):>5.0f}K: median {1000 * st.median(v):4.0f} ms (n={len(v):5d})"
              + (f"; full chunks {1000 * st.median(f):4.0f} ms (n={len(f)})" if f else ""))
