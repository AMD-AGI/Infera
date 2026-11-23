#!/usr/bin/env python3
"""Decode step decomposition from vpd_steplog CSVs (one file per DP rank process).
  steplog_summary.py STEPLOG_DIR [BENCHMARK_LOG | @T0:T1] [SKIP_S=300]
Window: AIPerf "Phase profiling (profiling) started" + SKIP_S .. "sending complete" in BENCHMARK_LOG, or epochs
T0 + SKIP_S .. T1, else everything.
Records of the DP ranks are matched into global steps by wall time (the ranks agree on the CUDA graph mode inside
execute_model, so they enter each step together). A step's duration is the time to the next step on rank 0 (lag 1;
lag 2 is printed too, since under async scheduling the cost of a step can surface one step later). Steps are classed:
  full   every rank ran a FULL graph
  new    not full; some rank scheduled a request with <= decode_query_len tokens that was new this step
         (a request entering decode after its remote KV: the last prompt token is recomputed)
  big    some rank scheduled a request with > 16 tokens (prefill-like: recompute after preemption, or a partial-block tail)
  other  not full, neither of the above
Gaps longer than 2 s (no work) are left out of step times.
"""
import collections
import datetime as dt
import glob
import os
import statistics as st
import sys

F = ["kind", "wall", "dt_ms", "cpu_ms", "rank", "dummy", "reqs", "toks", "maxq", "n_dec", "n_small", "n_big",
     "big_toks", "n_new", "n_new_small", "new_big_toks", "n_preempted", "mode", "desc_toks", "desc_reqs", "uniform"]
M = ["kind", "wall", "rank", "free_b", "total_b", "reserved_b", "allocated_b", "max_reserved_b"]


def window(log, skip):
    if log and log.startswith("@"):
        a, b = log[1:].split(":")
        return float(a) + skip, float(b)
    if not log or not os.path.exists(log):
        return None, None
    day = dt.datetime.fromtimestamp(os.path.getmtime(log), dt.timezone.utc).date()
    t0 = t1 = None
    for line in open(log, errors="replace"):
        if "Phase profiling (profiling) started" in line:
            t0 = line[:12]
        elif "Phase profiling (profiling) sending complete" in line:
            t1 = line[:12]

    def ts(hms):
        h, m, s = hms.split(":")
        return dt.datetime(day.year, day.month, day.day, int(h), int(m), int(float(s)), tzinfo=dt.timezone.utc).timestamp()

    return (ts(t0) + skip if t0 else None), (ts(t1) if t1 else None)


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))] if xs else float("nan")


skipped = [0]


def load(d):
    steps, mem = collections.defaultdict(list), collections.defaultdict(list)
    for path in sorted(glob.glob(os.path.join(d, "steps-r*.csv"))):
        for line in open(path, errors="replace"):
            if line.startswith("S,"):
                p = line.rstrip("\n").split(",")
                if len(p) != len(F):
                    continue
                r = dict(zip(F, p))
                for k in ("wall", "dt_ms", "cpu_ms"):
                    r[k] = float(r[k])
                for k in ("rank", "dummy", "reqs", "toks", "maxq", "n_dec", "n_small", "n_big", "big_toks", "n_new",
                          "n_new_small", "new_big_toks", "n_preempted"):
                    r[k] = int(r[k])
                if r["mode"] == "-":
                    # execute_model returned before dispatching (no tokens on this rank, e.g. a connector-only call):
                    # not a GPU step.
                    skipped[0] += 1
                    continue
                steps[r["rank"]].append(r)
            elif line.startswith("M,"):
                p = line.rstrip("\n").split(",")
                if len(p) == len(M):
                    r = dict(zip(M, p))
                    mem[int(r["rank"])].append({k: float(v) for k, v in r.items() if k not in ("kind", "rank")})
    for k in steps:
        steps[k].sort(key=lambda r: r["wall"])
    return steps, mem


def main():
    d = sys.argv[1]
    log = sys.argv[2] if len(sys.argv) > 2 else None
    skip = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    t0, t1 = window(log, skip)
    steps, mem = load(d)
    if not steps:
        print("no steplog records in", d)
        return
    ranks = sorted(steps)
    lo, hi = (t0 or -1e18), (t1 or 1e18)
    print(f"ranks {ranks}; window {'%.0f' % t0 if t0 else 'all'} .. {'%.0f' % t1 if t1 else 'all'} "
          f"({(hi - lo) if t0 and t1 else float('nan'):.0f} s); {skipped[0]} no-dispatch records (0 tokens) left out")

    # Per rank.
    print("\n== per rank (records in window)")
    for k in ranks:
        rs = [r for r in steps[k] if lo <= r["wall"] <= hi]
        modes = collections.Counter(r["mode"] for r in rs)
        dummy = sum(r["dummy"] for r in rs)
        pre = sum(r["n_preempted"] for r in rs)
        big = sum(r["big_toks"] for r in rs if not r["dummy"])
        cpu = {m: st.median([r["cpu_ms"] for r in rs if r["mode"] == m]) for m in modes}
        print(f"rank {k}: {len(rs)} steps, dummy {dummy}; modes " +
              ", ".join(f"{m} {n} ({100 * n / max(len(rs), 1):.1f}%, cpu p50 {cpu[m]:.1f} ms)" for m, n in modes.most_common()) +
              f"; preempted {pre}; >16-token request tokens {big:,}; reqs p50 {q([r['reqs'] for r in rs if not r['dummy']], .5)}")

    # Global steps anchored on the first rank.
    base = [r for r in steps[ranks[0]] if lo <= r["wall"] <= hi]
    if len(base) < 3:
        print("too few steps")
        return
    med_gap = st.median([b["wall"] - a["wall"] for a, b in zip(base, base[1:])])
    tol = max(0.5 * med_gap, 0.005)
    ptr = {k: 0 for k in ranks}
    glob_steps = []
    for b in base:
        grp = {ranks[0]: b}
        for k in ranks[1:]:
            rs = steps[k]
            i = ptr[k]
            while i + 1 < len(rs) and abs(rs[i + 1]["wall"] - b["wall"]) <= abs(rs[i]["wall"] - b["wall"]):
                i += 1
            ptr[k] = i
            if rs and abs(rs[i]["wall"] - b["wall"]) <= tol:
                grp[k] = rs[i]
        glob_steps.append(grp)
    matched = sum(len(g) == len(ranks) for g in glob_steps)
    print(f"\n== global steps: {len(glob_steps)} anchored on rank {ranks[0]}, fully matched {matched} "
          f"({100 * matched / len(glob_steps):.1f}%), median gap {med_gap * 1e3:.1f} ms, match tolerance {tol * 1e3:.1f} ms")

    # A rank running a prefill-like chunk can enter execute_model up to ~2 s away from the others (a0 interim: median
    # 1.45 s earlier), so it is often not matched into the group; look for its chunk on the unmatched ranks.
    import bisect
    big_walls = {k: [r["wall"] for r in steps[k] if r["n_big"] > 0 and not r["dummy"] and r["mode"] != "FULL"] for k in ranks}

    def big_nearby(k, t):
        ws = big_walls[k]
        i = bisect.bisect_left(ws, t - 2.0)
        return i < len(ws) and ws[i] <= t + 1.0

    def cls(g):
        rs = list(g.values())
        if all(r["mode"] == "FULL" for r in rs):
            return "full"
        if any(r["n_big"] > 0 and not r["dummy"] for r in rs):
            return "big"
        t = g[ranks[0]]["wall"]
        if any(big_nearby(k, t) for k in ranks if k not in g):
            return "big"
        if any(r["n_new_small"] > 0 and not r["dummy"] for r in rs):
            return "new"
        return "other"

    def busiest(g):
        return max((r["reqs"] for r in g.values() if not r["dummy"]), default=0)

    for lag in (1, 2):
        w = [g[ranks[0]]["wall"] for g in glob_steps]
        # lag 1: from this step to the next; lag 2: the step after this one.
        dur = [(w[i + lag] - w[i + lag - 1]) * 1e3 for i in range(len(w) - lag)]
        rows = [(cls(g), busiest(g), x) for g, x in zip(glob_steps, dur) if x < 2000]
        gaps = sum(1 for x in dur if x >= 2000)
        full_by_b = collections.defaultdict(list)
        for c, bz, x in rows:
            if c == "full":
                full_by_b[bz].append(x)
        full_med_all = st.median([x for c, _, x in rows if c == "full"]) if any(c == "full" for c, _, _ in rows) else float("nan")
        ref = {bz: st.median(v) for bz, v in full_by_b.items() if len(v) >= 20}

        def baseline(bz):
            if bz in ref:
                return ref[bz]
            near = sorted(ref, key=lambda b: abs(b - bz))
            return ref[near[0]] if near else full_med_all

        tot = sum(x for _, _, x in rows)
        print(f"\n== step classes, duration = {'time to the next step' if lag == 1 else 'the step after (lag 2)'} "
              f"({len(rows)} steps, {tot / 1e3:.0f} s; {gaps} gaps > 2 s left out)")
        print(f"{'class':>6} {'steps':>7} {'%steps':>7} {'time s':>8} {'%time':>6} {'p50 ms':>7} {'p90 ms':>7} {'excess s':>9} {'%time':>6}")
        excess_all = 0.0
        for c in ("full", "new", "big", "other"):
            xs = [x for cc, _, x in rows if cc == c]
            if not xs:
                continue
            ex = sum(max(0.0, x - baseline(bz)) for cc, bz, x in rows if cc == c) if c != "full" else 0.0
            excess_all += ex
            print(f"{c:>6} {len(xs):7d} {100 * len(xs) / len(rows):6.1f}% {sum(xs) / 1e3:8.1f} {100 * sum(xs) / tot:5.1f}% "
                  f"{q(xs, .5):7.1f} {q(xs, .9):7.1f} {ex / 1e3:9.1f} {100 * ex / tot:5.1f}%")
        print(f"excess over FULL steps of the same busiest-rank size: {excess_all / 1e3:.1f} s = {100 * excess_all / tot:.1f}% of step time "
              f"(mean step {tot / len(rows):.1f} ms, FULL median {full_med_all:.1f} ms)")
        if lag == 1:
            print("FULL step p50 ms by busiest-rank requests (>= 20 steps): " +
                  ", ".join(f"{bz}:{ref[bz]:.1f}" for bz in sorted(ref)))

    # Recompute after preemption: per rank, steps with > 16-token requests and their total.
    print("\n== prefill-like work on the decode (requests > 16 tokens)")
    for k in ranks:
        rs = [r for r in steps[k] if lo <= r["wall"] <= hi and not r["dummy"]]
        bigs = [r for r in rs if r["n_big"] > 0]
        news = [r for r in rs if r["new_big_toks"] > 0]
        print(f"rank {k}: {len(bigs)} steps, {sum(r['big_toks'] for r in bigs):,} tokens "
              f"(of which new-request tokens {sum(r['new_big_toks'] for r in news):,}); preempted {sum(r['n_preempted'] for r in rs)}; "
              f"max request tokens in a step {max((r['maxq'] for r in rs), default=0)}")

    # Memory.
    print("\n== memory (GiB)")
    G = 1 << 30
    for k in sorted(mem):
        ms = [m for m in mem[k] if lo <= m["wall"] <= hi] or mem[k]
        print(f"rank {k}: {len(ms)} samples; free min {min(m['free_b'] for m in ms) / G:.2f} p50 {q([m['free_b'] for m in ms], .5) / G:.2f}; "
              f"torch reserved max {max(m['reserved_b'] for m in ms) / G:.2f}, allocated max {max(m['allocated_b'] for m in ms) / G:.2f}, "
              f"max_reserved {max(m['max_reserved_b'] for m in ms) / G:.2f}; total {ms[0]['total_b'] / G:.1f}")


if __name__ == "__main__":
    main()
