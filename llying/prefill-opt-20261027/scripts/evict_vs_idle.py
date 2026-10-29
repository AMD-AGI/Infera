"""How much of the PP0 prefill ring time the device evictions cost, and how many of them an idle-time eviction (E8)
could have pre-empted, from a prefill log (PP0 "Prefill batch" lines: token usage, queue length).
An eviction shows as a drop of token usage >= --drop between consecutive batches. For each one: the batch gap around it,
the time since the last batch that saw an empty queue (an idle chance), and the device KV consumed since then
(usage rise x pool tokens), i.e. the free headroom an idle eviction would have needed to cover it.
  python3 evict_vs_idle.py PREFILL_LOG START_HH:MM:SS END_HH:MM:SS [--pool 10738496] [--drop 0.03]"""

import argparse
import re
import statistics as st
from datetime import datetime

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("start")
ap.add_argument("end")
ap.add_argument("--pool", type=int, default=10738496)
ap.add_argument("--drop", type=float, default=0.03)
a = ap.parse_args()

pat = re.compile(r"^(\S+)Z \[\S+ (\d\d:\d\d:\d\d) PP0\] Prefill batch.*?token usage: ([\d.]+).*?#queue-req: (\d+)")
rows = []
for line in open(a.log, errors="replace"):
    m = pat.search(line)
    if not m or not (a.start <= m.group(2) <= a.end):
        continue
    t = datetime.fromisoformat(m.group(1)[:26]).timestamp()
    rows.append((t, float(m.group(3)), int(m.group(4))))
gaps = [rows[i][0] - rows[i - 1][0] for i in range(1, len(rows))]
med_gap = st.median(gaps)
events, last_idle = [], None
for i in range(1, len(rows)):
    t, u, q = rows[i]
    if rows[i - 1][2] == 0:
        last_idle = i - 1
    if rows[i - 1][1] - u >= a.drop:
        since = (t - rows[last_idle][0]) if last_idle is not None else float("inf")
        used = (rows[i - 1][1] - rows[last_idle][1]) * a.pool if last_idle is not None else float("inf")
        events.append((t, gaps[i - 1], since, used))
span = rows[-1][0] - rows[0][0]
print(f"{len(rows)} PP0 batches over {span:.0f} s, median gap {med_gap * 1e3:.0f} ms; {len(events)} evictions "
      f"({len(events) / span * 60:.1f} / min)")
if events:
    ext = [g - med_gap for _, g, _, _ in events]
    print(f"PP0 batch gap at an eviction: median {st.median(g for _, g, _, _ in events) * 1e3:.0f} ms; "
          f"excess over the median gap summed {sum(ext):.0f} s = {sum(ext) / span:.1%} of the window (PP0 only; each rank repeats it)")
    for lim in (5, 10, 30, 60):
        n = sum(1 for _, _, s, _ in events if s <= lim)
        print(f"  evictions within {lim:3d} s after an empty-queue batch: {n} ({n / len(events):.0%})")
    for cap in (0.5e6, 1e6, 2e6, 3e6):
        n = sum(1 for _, _, _, u in events if u <= cap)
        print(f"  evictions whose KV use since the last idle batch <= {cap / 1e6:.1f}M tokens: {n} ({n / len(events):.0%})")
    idle = sum(1 for r in rows if r[2] == 0)
    print(f"batches seeing an empty queue: {idle} ({idle / len(rows):.0%})")

# Ring-wide view: all excess PP0 gap time (over the median gap) vs the part that falls within W s after an eviction
# (the other ranks repeat the eviction over the next beats, so PP0 waits in later gaps).
ev_t = [t for t, _, _, _ in events]
import bisect as _b
tot_ex = sum(max(0.0, g - med_gap) for g in gaps)
print(f"all excess gap time over the median: {tot_ex:.0f} s = {tot_ex / span:.1%} of the window")
for W in (2, 4, 6):
    ex_w = 0.0
    for i in range(1, len(rows)):
        g = gaps[i - 1]
        if g <= med_gap:
            continue
        t = rows[i][0]
        j = _b.bisect_right(ev_t, t) - 1
        if j >= 0 and t - ev_t[j] <= W:
            ex_w += g - med_gap
    print(f"  of which within {W} s after an eviction: {ex_w:.0f} s = {ex_w / span:.1%} of the window ({ex_w / tot_ex:.0%} of the excess)")
# control: the same windows placed at random times
import random as _r
_r.seed(0)
fake = sorted(_r.uniform(rows[0][0], rows[-1][0]) for _ in ev_t)
ex_f = 0.0
for i in range(1, len(rows)):
    g = gaps[i - 1]
    if g <= med_gap:
        continue
    t = rows[i][0]
    j = _b.bisect_right(fake, t) - 1
    if j >= 0 and t - fake[j] <= 4:
        ex_f += g - med_gap
print(f"  control: within 4 s after {len(fake)} random times: {ex_f:.0f} s ({ex_f / tot_ex:.0%} of the excess)")

# Only gaps where the previous batch still saw a queue (ring inefficiency, not lack of work).
busy = [(rows[i][0], gaps[i - 1]) for i in range(1, len(rows)) if rows[i - 1][2] > 0]
bmed = st.median(g for _, g in busy)
bex = sum(max(0.0, g - bmed) for _, g in busy)
print(f"queued gaps: {len(busy)}, median {bmed * 1e3:.0f} ms, excess over it {bex:.0f} s = {bex / span:.1%} of the window")
for name, ts in (("evictions", ev_t), ("random", fake)):
    ex = 0.0
    for t, g in busy:
        if g <= bmed:
            continue
        j = _b.bisect_right(ts, t) - 1
        if j >= 0 and t - ts[j] <= 4:
            ex += g - bmed
    print(f"  queued excess within 4 s after {name}: {ex:.0f} s ({ex / bex:.0%})")
