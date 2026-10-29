"""Which per-batch features go with long prefill beats (queued only), from the PP0 prefill log: beat = time to the next PP0
batch line when the queue was non-empty. For each feature, beats are split by the feature's quartiles and each bin reports
the median beat and the share of the total excess (beat over the overall median) -- a feature that drives long beats
concentrates the excess in one bin far beyond its 25 % share of batches.
  python3 beat_features.py PREFILL_LOG START_HH:MM:SS END_HH:MM:SS"""

import re
import statistics as st
import sys
from datetime import datetime

log, t0, t1 = sys.argv[1:4]
pat = re.compile(r"^(\S+)Z \[\S+ (\d\d:\d\d:\d\d) PP0\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+), "
                 r"token usage: ([\d.]+), #running-req: \d+, #queue-req: (\d+), #pending-token: (\d+), #bootstrap-req: (\d+), "
                 r"#inflight-req: (\d+)")
rows = []
for line in open(log, errors="replace"):
    m = pat.search(line)
    if m and t0 <= m.group(2) <= t1:
        t = datetime.fromisoformat(m.group(1)[:26]).timestamp()
        rows.append(dict(t=t, seqs=int(m.group(3)), new=int(m.group(4)), cached=int(m.group(5)), usage=float(m.group(6)),
                         queue=int(m.group(7)), pending=int(m.group(8)), bootstrap=int(m.group(9)), inflight=int(m.group(10))))
beats = []
for i in range(len(rows) - 1):
    if rows[i]["queue"] > 0:
        r = dict(rows[i])
        r["beat"] = rows[i + 1]["t"] - rows[i]["t"]
        beats.append(r)
med = st.median(r["beat"] for r in beats)
tot = sum(max(0.0, r["beat"] - med) for r in beats)
print(f"{len(beats)} queued beats, median {med * 1e3:.0f} ms, excess over median {tot:.0f} s")
for feat in ("cached", "new", "seqs", "queue", "pending", "bootstrap", "inflight", "usage"):
    vals = sorted(r[feat] for r in beats)
    qs = [vals[int(len(vals) * k / 4)] for k in (1, 2, 3)]
    bins = [[], [], [], []]
    for r in beats:
        b = sum(1 for q in qs if r[feat] > q)
        bins[b].append(r)
    parts = []
    for b, rs in enumerate(bins):
        if not rs:
            parts.append("  -")
            continue
        ex = sum(max(0.0, r["beat"] - med) for r in rs)
        parts.append(f"Q{b + 1} n{len(rs):5d} p50 {st.median(r['beat'] for r in rs) * 1e3:4.0f}ms ex {ex / tot:4.0%}")
    print(f"{feat:9s} cuts {qs}: " + " | ".join(parts))
