"""What one device eviction (evict_for_alloc) spends its time on, from compacted 4-rank traces (trace_compact.py):
  - scheduler-thread frames inside it, summed by name (outer frames include inner ones; read as a call tree),
  - HIP runtime calls inside it (syncs / copies) and the GPU copies it issued (D2H bytes, copy time on the GPU),
  - python frames of other threads that overlap it (GIL contention candidates), vs the same quantity in equally long
    windows right before each eviction (control).
  python3 evict_breakdown.py COMPACT_DIR [--top 25]"""

import argparse
import bisect
import collections
import gzip
import json
import re
import statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("compact")
ap.add_argument("--top", type=int, default=25)
a = ap.parse_args()

EV = re.compile(r"unified_radix_cache\.py\(\d+\): evict_for_alloc$")
short = lambda n: re.sub(r".*/(sglang/srt/|sglang/)", "", n)[:90]


def inside(rows, t0, t1, key=lambda r: r[1]):
    ts = [key(r) for r in rows]
    i = bisect.bisect_left(ts, t0)
    out = []
    while i < len(rows) and key(rows[i]) < t1:
        out.append(rows[i])
        i += 1
    return out


def overlap(rows, t0, t1):
    """Per thread: time inside [t0, t1] covered by at least one python frame (nested frames counted once)."""
    by = collections.defaultdict(list)
    for r in rows:
        s, e = max(t0, r[1]), min(t1, r[1] + r[2])
        if e > s:
            by[r[0]].append((s, e))
    tot = collections.Counter()
    for k, iv in by.items():
        iv.sort()
        cs, ce = iv[0]
        for s, e in iv[1:]:
            if s > ce:
                tot[k] += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        tot[k] += ce - cs
    return tot


for rk in range(4):
    R = json.load(gzip.open(f"{a.compact}/pp{rk}.json.gz", "rt"))
    sched = R["sched"]
    evs = [r for r in sched if r[0] == "p" and EV.search(r[3])]
    if not evs:
        continue
    frames = collections.Counter()
    rt = collections.Counter()
    rtn = collections.Counter()
    d2h_bytes = d2h_gpu = 0.0
    other_in = collections.Counter()
    other_ctrl = collections.Counter()
    for e in evs:
        t0, t1 = e[1], e[1] + e[2]
        seen = set()
        for r in inside(sched, t0, t1):
            if r[1] + r[2] > t1 + 1e-6:
                continue
            nm = short(r[3])
            if r[0] == "p":
                if (nm, round(r[1], 6)) not in seen:
                    frames[nm] += r[2]
                    seen.add((nm, round(r[1], 6)))
            elif r[0] == "c":
                rt[nm] += r[2]
                rtn[nm] += 1
        for c in inside(R["copies"], t0, t1, key=lambda c: c[0]):
            if "DtoH" in c[3] or "D2H" in c[3] or "Device -> Host" in c[3]:
                d2h_bytes += c[4] or 0
                d2h_gpu += c[1]
        other_in += overlap(R["others"], t0, t1)
        other_ctrl += overlap(R["others"], t0 - (t1 - t0), t0)
    tot = sum(e[2] for e in evs)
    print(f"=== PP{rk}: {len(evs)} evictions, {tot:.2f} s total, mean {tot / len(evs) * 1e3:.0f} ms "
          f"(min {min(e[2] for e in evs) * 1e3:.0f}, max {max(e[2] for e in evs) * 1e3:.0f})")
    print(f"  D2H copies issued inside: {d2h_bytes / 1e9 / len(evs):.2f} GB per eviction, GPU copy time "
          f"{d2h_gpu / len(evs) * 1e3:.0f} ms per eviction")
    print("  HIP runtime calls (>= 1 ms) inside, per eviction: " + "; ".join(
        f"{k} {v / len(evs) * 1e3:.0f} ms ({rtn[k] / len(evs):.0f}x)" for k, v in rt.most_common(6)))
    print("  scheduler frames inside (mean ms per eviction, nested):")
    for k, v in frames.most_common(a.top):
        print(f"    {v / len(evs) * 1e3:7.1f}  {k}")
    th = R["threads"]
    print("  other threads' python time overlapping the evictions vs the equally long window just before (s total):")
    for k in sorted(set(other_in) | set(other_ctrl), key=lambda k: -(other_in[k] + other_ctrl[k]))[:6]:
        print(f"    {k} {th.get(k, '?')}: during {other_in[k]:.2f}, before {other_ctrl[k]:.2f}")
