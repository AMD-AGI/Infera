#!/usr/bin/env python3
"""Decode step time against load, controlling for context size.

Each DP rank logs one "Decode batch" line every 40 steps; step time = (gap to the rank's next line) / 40. DP attention
steps all ranks together, so a step is paced by the busiest rank: within each 3-s bin take the max #token (KV tokens of
the running requests on a rank = context in its batch) and max #running-req over ranks, and the median step time of the
intervals that start in that bin. Prints median step ms by busiest-rank context bucket and by busiest-rank running count,
plus a least-squares fit step_ms = a + b * max_tokens(M) + c * max_running.

Usage: decode_pace2.py DECODE_LOG START_ISO END_ISO [LABEL]
"""

import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime

PAT = re.compile(r"^(\S+?)Z? \[[^\]]*DP(\d+) TP\d+\] Decode batch, #running-req: (\d+), #token: (\d+), "
                 r"token usage: ([0-9.]+), accept len: ([0-9.]+)")


def main():
    log, t0, t1 = sys.argv[1], datetime.fromisoformat(sys.argv[2]), datetime.fromisoformat(sys.argv[3])
    label = sys.argv[4] if len(sys.argv) > 4 else log
    rows = defaultdict(list)
    for line in open(log, errors="ignore"):
        m = PAT.match(line)
        if not m:
            continue
        t = datetime.fromisoformat(m.group(1)[:26])
        if t0 <= t <= t1:
            rows[int(m.group(2))].append(((t - t0).total_seconds(), int(m.group(3)), int(m.group(4)), float(m.group(6))))
    bins = defaultdict(lambda: {"tok": 0, "run": 0, "steps": [], "acc": []})
    for dp, r in rows.items():
        for a, b in zip(r, r[1:]):
            b_ = bins[int(a[0] // 3)]
            b_["tok"] = max(b_["tok"], a[2])
            b_["run"] = max(b_["run"], a[1])
            b_["acc"].append(a[3])
            gap = b[0] - a[0]
            if 0 < gap < 20:
                b_["steps"].append(gap / 40 * 1000)
    pts = [(v["tok"] / 1e6, v["run"], statistics.median(v["steps"])) for v in bins.values() if v["steps"] and v["run"] > 0]
    by_tok, by_run = defaultdict(list), defaultdict(list)
    for tok, run, ms in pts:
        by_tok[min(int(tok / 0.5), 7)].append(ms)
        by_run[min(run, 16)].append(ms)
    print(f"== {label}: {len(pts)} bins; busiest-rank context mean {statistics.mean(p[0] for p in pts):.2f}M, "
          f"running mean {statistics.mean(p[1] for p in pts):.1f}")
    print("  step ms by busiest-rank context: " + "  ".join(
        f"{k * 0.5:.1f}-{k * 0.5 + 0.5:.1f}M:{statistics.median(v):5.1f}[{len(v)}]" for k, v in sorted(by_tok.items()) if len(v) >= 5))
    print("  step ms by busiest-rank running: " + "  ".join(
        f"{k}:{statistics.median(v):5.1f}[{len(v)}]" for k, v in sorted(by_run.items()) if len(v) >= 5 and k >= 4))
    # least squares a + b*tok + c*run
    n = len(pts)
    if n > 10:
        X = [(1.0, p[0], float(p[1])) for p in pts]
        y = [p[2] for p in pts]
        XtX = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
        Xty = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(3)]
        # solve 3x3
        import copy
        A = copy.deepcopy(XtX)
        bvec = Xty[:]
        for i in range(3):
            piv = A[i][i]
            for j in range(i, 3):
                A[i][j] /= piv
            bvec[i] /= piv
            for k in range(3):
                if k != i:
                    f = A[k][i]
                    for j in range(i, 3):
                        A[k][j] -= f * A[i][j]
                    bvec[k] -= f * bvec[i]
        a, b, c = bvec
        print(f"  fit: step_ms = {a:.1f} + {b:.1f} * context_M + {c:.2f} * running")


if __name__ == "__main__":
    main()
