"""Whole-run test: do the long queued PP0 prefill beats coincide with real device evictions (any PP rank), and does
token usage still predict long beats once evictions are accounted for?
Evictions come from aiperf's server-metrics export (1 s slices of sglang:eviction_duration_seconds per PP rank, scraped
every ~333 ms), not from token-usage drops (an eviction moves evictable -> free and leaves usage unchanged).
Each slice with c evictions of total duration S is taken as busy over [slice_end - S - lag_slack, slice_end].
Control: the same eviction intervals circularly shifted by random offsets (keeps their rate and clustering).
  python3 beat_vs_evict.py PREFILL_LOG SERVER_METRICS_EXPORT.json [--endpoint 27201] [--margin 0.5] [--shifts 40]"""

import argparse
import bisect
import json
import random
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from beat_usage_window import load  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("log")
ap.add_argument("metrics")
ap.add_argument("--endpoint", default="27201")
ap.add_argument("--margin", type=float, default=0.5, help="widen each eviction interval by this much on both sides (s)")
ap.add_argument("--shifts", type=int, default=40)
a = ap.parse_args()

d = json.load(open(a.metrics))
M = d["metrics"]
ph = d["summary"]["phase_time_ranges"]["profiling"]
w0, w1 = ph["start_ns"] / 1e9, ph["end_ns"] / 1e9


def series(name, rank):
    for s in M[name]["series"]:
        if a.endpoint in s["endpoint_url"] and s["labels"].get("pp_rank") == str(rank):
            return s["timeslices"]
    raise KeyError((name, rank))


# --- clock check: PP0 log usage vs metrics token_usage (per-second means), best offset
rows_all = load(a.log, None)
rows = [r for r in rows_all if r["rank"] == 0 and w0 <= r["t"] <= w1]
lu = defaultdict(list)
for r in rows:
    lu[int(r["t"])].append(r["usage"])
mu = {int(x["start_ns"] / 1e9): x["avg"] for x in series("sglang:token_usage", 0) if "avg" in x}
best = []
for off in range(-4, 5):
    pairs = [(st.mean(v), mu[s + off]) for s, v in lu.items() if s + off in mu]
    if len(pairs) > 100:
        xs, ys = zip(*pairs)
        best.append((st.correlation(xs, ys), off, len(pairs)))
best.sort(reverse=True)
print("clock check (corr of PP0 log usage vs metrics usage at slice offset s): "
      + ", ".join(f"{o:+d}: {c:.3f}" for c, o, _ in sorted(best, key=lambda x: x[1])))

# --- eviction intervals per rank
ev = {}
for rk in range(4):
    iv = []
    for x in series("sglang:eviction_duration_seconds", rk):
        if x["count"]:
            e = x["end_ns"] / 1e9
            iv.append((e - x["sum"] - a.margin, e + a.margin, x["count"], x["sum"]))
    ev[rk] = iv
    print(f"PP{rk}: {sum(i[2] for i in iv)} evictions, {sum(i[3] for i in iv):.0f} s inside evict")
lb = {}
for rk in range(4):
    lb[rk] = {int(x["start_ns"] / 1e9): (x["count"], x["sum"]) for x in series("sglang:load_back_duration_seconds", rk)}
lbt = {int(x["start_ns"] / 1e9): x["total"] for x in series("sglang:load_back_tokens", 0)}


def merged(ivs):
    out = []
    for s, e in sorted(ivs):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def shift(ivs, off):
    span = w1 - w0
    out = []
    for s, e in ivs:
        s2 = (s - w0 + off) % span + w0
        out.append((s2, s2 + (e - s)))
    return out


def overlaps(m, t0, t1):
    starts = [x[0] for x in m]
    j = bisect.bisect_right(starts, t1) - 1
    while j >= 0 and m[j][1] >= t0 - 30:
        if m[j][0] <= t1 and m[j][1] >= t0:
            return True
        j -= 1
    return False


# --- PP0 queued beats
beats = []
for i in range(len(rows) - 1):
    if rows[i]["queue"] > 0:
        beats.append(dict(t0=rows[i]["t"], t1=rows[i + 1]["t"], g=rows[i + 1]["t"] - rows[i]["t"], u=rows[i]["usage"],
                          seqs=rows[i]["seqs"], cached=rows[i]["cached"]))
med = st.median(b["g"] for b in beats)
for b in beats:
    b["ex"] = max(0.0, b["g"] - med)
tot = sum(b["ex"] for b in beats)
span = w1 - w0
print(f"window {span:.0f} s; {len(beats)} queued PP0 beats, median {med * 1e3:.0f} ms, excess {tot:.0f} s = {tot / span:.1%} of window")

any_iv = merged([(s, e) for rk in range(4) for s, e, _, _ in ev[rk]])
cover = sum(e - s for s, e in any_iv)
print(f"eviction intervals (any rank, margin {a.margin} s) cover {cover:.0f} s = {cover / span:.0%} of the window")


def split(m):
    hit = [b for b in beats if overlaps(m, b["t0"], b["t1"])]
    return len(hit) / len(beats), sum(b["ex"] for b in hit) / tot, hit


fb, fe, hit = split(any_iv)
random.seed(1)
ctrl = []
for _ in range(a.shifts):
    off = random.uniform(30, span - 30)
    cb, ce, _ = split(merged(shift([(s, e) for s, e in any_iv], off)))
    ctrl.append((cb, ce))
print(f"beats overlapping an eviction (any rank): {fb:.0%} of beats carry {fe:.0%} of the excess; "
      f"shifted control: {st.mean(c[0] for c in ctrl):.0%} of beats carry {st.mean(c[1] for c in ctrl):.0%} "
      f"(sd {st.pstdev(c[1] for c in ctrl):.1%}, max {max(c[1] for c in ctrl):.0%})")
hs = set(id(b) for b in hit)
for name, bs in (("with eviction", [b for b in beats if id(b) in hs]), ("without", [b for b in beats if id(b) not in hs])):
    if bs:
        print(f"  {name:13s}: n {len(bs):5d}, beat p50 {st.median(b['g'] for b in bs) * 1e3:4.0f} ms, "
              f"p90 {sorted(b['g'] for b in bs)[int(0.9 * len(bs))] * 1e3:5.0f} ms, excess {sum(b['ex'] for b in bs):5.0f} s")
wo = [b for b in beats if id(b) not in hs]
if wo:
    base = st.mean(b["ex"] for b in wo)
    attr = sum(b["ex"] for b in hit) - len(hit) * base
    print(f"  excess beyond what the same beats would carry at the no-eviction rate ({base * 1e3:.0f} ms / beat): "
          f"{attr:.0f} s = {attr / tot:.0%} of the excess = {attr / span:.1%} of the window")
print("lag scan (eviction intervals shifted by LAG s; share of beats hit -> share of excess):")
for lag in (-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3):
    lb_, le_, _ = split(merged(shift([(s, e) for s, e in any_iv], lag)))
    print(f"  lag {lag:+4.1f}: {lb_:.0%} -> {le_:.0%} (ratio {le_ / lb_:.2f})")

# The old detector (evict_vs_idle.py): PP0 usage drop >= 0.03 between consecutive batches taken as an eviction.
drops = [rows[i]["t"] for i in range(1, len(rows)) if rows[i - 1]["usage"] - rows[i]["usage"] >= 0.03]
pp0_iv = merged([(s, e) for s, e, _, _ in ev[0]])
in_pp0 = sum(1 for t in drops if overlaps(pp0_iv, t - 0.5, t + 0.5))
cov0 = sum(e - s for s, e in merged([(s - 0.5, e + 0.5) for s, e in pp0_iv])) / span
print(f"old detector: {len(drops)} usage drops >= 0.03 (real PP0 evictions: {sum(i[2] for i in ev[0])}); "
      f"{in_pp0 / max(len(drops), 1):.0%} fall within 0.5 s of a real PP0 eviction interval, "
      f"which (widened) cover {cov0:.0%} of the window")
for rk in range(4):
    fb_r, fe_r, _ = split(merged([(s, e) for s, e, _, _ in ev[rk]]))
    print(f"  only PP{rk} evictions: {fb_r:.0%} of beats carry {fe_r:.0%} of the excess")

# --- usage quartiles x eviction overlap
us = sorted(b["u"] for b in beats)
qs = [us[int(len(us) * k / 4)] for k in (1, 2, 3)]
print(f"usage quartile cuts {qs}")
print("  usage bin | beats  share-of-excess | with eviction: n, p50, excess | without: n, p50, excess")
for q in range(4):
    bs = [b for b in beats if sum(1 for c in qs if b["u"] > c) == q]
    w = [b for b in bs if id(b) in hs]
    wo = [b for b in bs if id(b) not in hs]
    f = lambda xs: (f"{len(xs):5d} {st.median(b['g'] for b in xs) * 1e3:4.0f}ms {sum(b['ex'] for b in xs):5.0f}s" if xs else "    -")
    print(f"  Q{q + 1}        | {len(bs):5d} {sum(b['ex'] for b in bs) / tot:5.0%}           | {f(w)}          | {f(wo)}")

# --- per second (metrics only): does usage go with eviction / loadback rate?
evc0 = defaultdict(int)
for x in series("sglang:eviction_duration_seconds", 0):
    evc0[int(x["start_ns"] / 1e9)] += x["count"]
secs = [s for s in mu if w0 <= s <= w1]
su = sorted(mu[s] for s in secs)
sq = [su[int(len(su) * k / 4)] for k in (1, 2, 3)]
print(f"per second, metrics usage quartile cuts {sq}: PP0 evictions / s, loadback tokens / s (PP0)")
for q in range(4):
    ss = [s for s in secs if sum(1 for c in sq if mu[s] > c) == q]
    print(f"  Q{q + 1}: {len(ss)} s, evictions {sum(evc0[s] for s in ss) / len(ss):.2f} / s, "
          f"loadback {st.mean(lbt.get(s, 0.0) for s in ss):,.0f} tok / s")
