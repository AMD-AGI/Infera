#!/usr/bin/env python3
"""What sets a DP-attention decode step: the busiest rank's request count or its context tokens?

Every rank logs `Decode batch, #running-req: R, #token: T` once per 40 steps; ranks step in lockstep, so the
step time is (interval between a rank's consecutive lines) / 40. Per 3 s bin this takes each rank's latest R and T,
the median step time, and fits by least squares
    step_ms = a + b*max_r(R_r) + c*max_r(T_r)/1e6 + d*sum_r(R_r) + e*sum_r(T_r)/1e6
then reports b / c: how many tokens on the busiest rank cost as much as one more request there (the per-request
weight a request-count balancer should use).
Usage: decode_step_fit.py DECODE_LOG START_ISO END_ISO
"""

import re
import statistics
import sys
from datetime import datetime

LINE = re.compile(r"^(\S+)Z \[[^\]]*DP(\d+) TP\d+\] Decode batch, #running-req: (\d+), #token: (\d+)")
INTERVAL = 40
BIN_S = 3.0


def ts(text):
    return datetime.fromisoformat(text[:26])


def solve(rows, ys):
    """Least squares via normal equations (no numpy on this host)."""
    k = len(rows[0])
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(k)] for i in range(k)]
    aty = [sum(r[i] * y for r, y in zip(rows, ys)) for i in range(k)]
    m = [ata[i] + [aty[i]] for i in range(k)]
    for c in range(k):
        p = max(range(c, k), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(k):
            if r != c and m[c][c]:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][k] / m[i][i] for i in range(k)]


def main():
    path, start, end = sys.argv[1], ts(sys.argv[2]), ts(sys.argv[3])
    last, state, bins = {}, {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            m = LINE.match(line)
            if not m:
                continue
            t = ts(m.group(1))
            if not start <= t <= end:
                continue
            dp, run, tok = int(m.group(2)), int(m.group(3)), int(m.group(4))
            if dp in last:
                step = (t - last[dp]).total_seconds() * 1000 / INTERVAL
                if 5 < step < 500:
                    b = int((t - start).total_seconds() // BIN_S)
                    bins.setdefault(b, {"steps": [], "state": None})["steps"].append(step)
            last[dp] = t
            state[dp] = (run, tok)
            if len(state) == 8:
                b = int((t - start).total_seconds() // BIN_S)
                bins.setdefault(b, {"steps": [], "state": None})["state"] = dict(state)
    rows, ys, ratio, busiest, mean_r, tok_ratio = [], [], [], [], [], []
    for b in bins.values():
        if not b["steps"] or not b["state"]:
            continue
        rs = [v[0] for v in b["state"].values()]
        ts_ = [v[1] / 1e6 for v in b["state"].values()]
        rows.append([1.0, max(rs), max(ts_), sum(rs), sum(ts_)])
        ys.append(statistics.median(b["steps"]))
        busiest.append(max(rs))
        mean_r.append(sum(rs) / 8)
        if sum(rs):
            ratio.append(max(rs) / (sum(rs) / 8))
        if sum(ts_):
            tok_ratio.append(max(ts_) / (sum(ts_) / 8))
    a, bb, c, d, e = solve(rows, ys)
    pred = [sum(x * w for x, w in zip(r, (a, bb, c, d, e))) for r in rows]
    resid = [y - p for y, p in zip(ys, pred)]
    ss_res = sum(r * r for r in resid)
    mean = statistics.mean(ys)
    ss_tot = sum((y - mean) ** 2 for y in ys)
    print(f"bins {len(rows)}  step median {statistics.median(ys):.1f} ms  R^2 {1 - ss_res / ss_tot:.3f}  "
          f"rmse {(ss_res / len(rows)) ** 0.5:.2f} ms")
    print(f"step_ms = {a:.1f} + {bb:.2f}*max_req + {c:.2f}*max_tok(M) + {d:.3f}*sum_req + {e:.3f}*sum_tok(M)")
    if c > 0:
        print(f"busiest-rank tokens equivalent to one request there: {bb / c * 1e6:,.0f}")
    p90 = lambda xs: sorted(xs)[int(len(xs) * 0.9)]  # noqa: E731
    print(f"running per rank: mean {statistics.mean(mean_r):.2f}  busiest median {statistics.median(busiest):.0f}  "
          f"p90 {p90(busiest):.0f}")
    print(f"busiest / mean running per rank: median {statistics.median(ratio):.2f}  p90 {p90(ratio):.2f}")
    print(f"busiest / mean context per rank: median {statistics.median(tok_ratio):.2f}  p90 {p90(tok_ratio):.2f}")
    # Single-term fits, to see which busiest-rank quantity explains the step on its own.
    for name, idx in (("max_req", 1), ("max_tok", 2)):
        sub = [[1.0, r[idx], r[3], r[4]] for r in rows]
        w = solve(sub, ys)
        p = [sum(x * k for x, k in zip(r, w)) for r in sub]
        ss = sum((y - q) ** 2 for y, q in zip(ys, p))
        print(f"  only {name} + totals: R^2 {1 - ss / ss_tot:.3f}  coef {w[1]:.2f}")


if __name__ == "__main__":
    main()
