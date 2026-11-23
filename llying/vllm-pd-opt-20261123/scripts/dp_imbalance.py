#!/usr/bin/env python3
"""Cost of decode DP imbalance from a steplog directory (instr/vpd_steplog.py CSVs).
  dp_imbalance.py STEPLOG_DIR [A_REQ=0.74] [B_CTX_PER_M=3.9] [T0 T1]
Matches the ranks' FULL records into global steps (nearest record within 23 ms of rank 0's), and per step computes the
per-rank cost c = A_REQ * reqs + B_CTX_PER_M * ctx_sum / 1e6 (the a3 step fit's coefficients). Reports the mean of
max(c) - mean(c) (what a perfect balancer could save per step), its split into the request and context parts, and how
often the busiest rank by cost is also the busiest by requests / by context.
"""
import bisect
import glob
import os
import sys


def load(d, t0, t1):
    ranks = {}
    for f in glob.glob(os.path.join(d, "steps-r*.csv")):
        for line in open(f):
            if not line.startswith("S,"):
                continue
            p = line.rstrip("\n").split(",")
            # S,wall,dt_ms,cpu_ms,rank,dummy,reqs,toks,...,mode(17),desc_toks,desc_reqs,uniform,ctx_sum(21),ctx_max
            if len(p) < 23 or p[17] != "FULL":
                continue
            wall = float(p[1])
            if t0 is not None and not (t0 <= wall <= t1):
                continue
            ranks.setdefault(int(p[4]), []).append((wall, int(p[6]), int(p[21])))
    for v in ranks.values():
        v.sort()
    return ranks


def main():
    d = sys.argv[1]
    a = float(sys.argv[2]) if len(sys.argv) > 2 else 0.74
    b = float(sys.argv[3]) if len(sys.argv) > 3 else 3.9
    t0, t1 = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (None, None)
    ranks = load(d, t0, t1)
    if 0 not in ranks or len(ranks) < 2:
        sys.exit(f"need rank 0 and others in {d}")
    walls = {r: [x[0] for x in v] for r, v in ranks.items()}
    n = gap = gap_req = gap_ctx = 0.0
    same_req = same_ctx = 0
    spread_req = spread_ctx = 0.0
    for w, q0, c0 in ranks[0]:
        rows = [(q0, c0)]
        for r, v in ranks.items():
            if r == 0:
                continue
            i = bisect.bisect_left(walls[r], w)
            best = None
            for j in (i - 1, i):
                if 0 <= j < len(v) and abs(v[j][0] - w) < 0.023 and (best is None or abs(v[j][0] - w) < abs(v[best][0] - w)):
                    best = j
            if best is None:
                break
            rows.append(v[best][1:])
        if len(rows) != len(ranks):
            continue
        cost = [a * q + b * c / 1e6 for q, c in rows]
        k = max(range(len(rows)), key=lambda i: cost[i])
        mq = sum(q for q, _ in rows) / len(rows)
        mc = sum(c for _, c in rows) / len(rows)
        n += 1
        gap += cost[k] - sum(cost) / len(cost)
        gap_req += a * (rows[k][0] - mq)
        gap_ctx += b * (rows[k][1] - mc) / 1e6
        spread_req += max(q for q, _ in rows) - mq
        spread_ctx += (max(c for _, c in rows) - mc) / 1e6
        same_req += rows[k][0] == max(q for q, _ in rows)
        same_ctx += rows[k][1] == max(c for _, c in rows)
    if not n:
        sys.exit("no matched steps")
    print(f"{d}: {int(n)} matched FULL steps over {len(ranks)} ranks; cost = {a} * reqs + {b} * ctx(M)")
    print(f"  busiest-by-cost minus mean: {gap / n:.2f} ms per step (requests part {gap_req / n:.2f}, context part {gap_ctx / n:.2f})")
    print(f"  max - mean per step: requests {spread_req / n:.2f}, context {spread_ctx / n:.3f} M")
    print(f"  busiest by cost is also busiest by requests {same_req / n:.0%}, by context {same_ctx / n:.0%}")


if __name__ == "__main__":
    main()
