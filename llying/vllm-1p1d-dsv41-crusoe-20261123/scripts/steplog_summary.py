#!/usr/bin/env python3
"""Per-step summary of one leg's vpd_steplog CSVs (steps-r<rank>-<pid>.csv) inside a time window.

  steplog_summary.py STEPLOG_DIR ROLE T0 T1        (T0 / T1: unix seconds, e.g. the AgentX profiling window)

Per DP rank: step count and rate, cadence (dt_ms) mean / p50 / p90, scheduled tokens and requests per step,
real vs dummy steps, CUDA-graph mode mix. Decode also gets the least-squares fit dt = a + b * reqs over real
steps, and the share of steps wider than the largest captured FULL decode graph (desc_reqs empty or mode NONE).
"""
import csv
import glob
import os
import statistics
import sys
from collections import Counter


def pct(v, q):
    if not v:
        return float("nan")
    s = sorted(v)
    return s[min(len(s) - 1, int(q * len(s)))]


def main():
    d, role, t0, t1 = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])
    rows = {}
    for path in sorted(glob.glob(os.path.join(d, "steps-r*-*.csv"))):
        with open(path) as f:
            for r in csv.reader(f):
                if not r or r[0] != "S" or len(r) < 23:
                    continue
                try:
                    wall = float(r[1])
                except ValueError:
                    continue
                if t0 <= wall <= t1:
                    rows.setdefault(int(r[4]), []).append(r)
    span = t1 - t0
    print(f"{role}: window {span:.0f} s, files {len(glob.glob(os.path.join(d, 'steps-r*-*.csv')))}")
    for rank in sorted(rows):
        rs = rows[rank]
        real = [r for r in rs if r[5] == "0"]
        dt = [float(r[2]) for r in real if float(r[2]) > 0]
        toks = [int(r[7]) for r in real]
        reqs = [int(r[6]) for r in real]
        modes = Counter(r[17] for r in real)
        print(f"  rank {rank}: steps {len(rs)} (real {len(real)}, dummy {len(rs) - len(real)}), "
              f"{len(rs) / span:.2f} steps/s")
        if dt:
            print(f"    cadence ms: mean {statistics.mean(dt):.1f} p50 {pct(dt, .5):.1f} p90 {pct(dt, .9):.1f}")
        if toks:
            print(f"    tokens/step: mean {statistics.mean(toks):.0f} p50 {pct(toks, .5):.0f} p90 {pct(toks, .9):.0f}"
                  f"  -> {sum(toks) / span:.0f} tok/s; reqs/step mean {statistics.mean(reqs):.1f}")
        print(f"    graph mode: {dict(modes.most_common())}")
        if role.upper().startswith("D") and len(real) > 100:
            pts = [(int(r[6]), float(r[2])) for r in real if float(r[2]) > 0 and int(r[13]) == 0]
            if len(pts) > 100:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                mx, my = statistics.mean(xs), statistics.mean(ys)
                sxx = sum((x - mx) ** 2 for x in xs)
                b = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else float("nan")
                a = my - b * mx
                print(f"    decode-only steps {len(pts)}: dt = {a:.1f} ms + {b:.3f} ms x reqs (mean reqs {mx:.0f}, "
                      f"mean dt {my:.1f} ms)")
            wide = sum(1 for r in real if r[19] == "" or r[17] in ("NONE", "-"))
            print(f"    steps without a FULL graph descriptor: {wide} of {len(real)} ({100 * wide / len(real):.0f}%)")


if __name__ == "__main__":
    main()
