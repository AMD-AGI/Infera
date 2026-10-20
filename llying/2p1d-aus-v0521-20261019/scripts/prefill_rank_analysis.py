#!/usr/bin/env python3
"""Per-DP-rank prefill load and head-of-line blocking over a window, from the prefill server logs.

Per rank: batches, new tokens (and rate), busy share (intervals to a next batch starting within 3 s), backlogged share,
queue mean / p90, and runs of consecutive full single-request chunks (one long request occupying the rank).
Across ranks, second by second: share of time some rank is idle while another has >= 5 queued, and the queue spread.
Overall: share of prefill busy time and new tokens spent in full single-request chunks.

Usage: prefill_rank_analysis.py SERVER_LOG_DIR START_ISO END_ISO [INSTANCE ...]   (default prefill-0 prefill-1)
"""

import bisect
import re
import sys
from collections import defaultdict
from datetime import datetime

PAT = re.compile(r"^(\S+) \[\S+ \S+ DP(\d) TP\d\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), "
                 r"#cached-token: (\d+), token usage: [0-9.]+, #running-req: \d+, #queue-req: (\d+), #pending-token: (\d+)")


def main():
    logs, t0, t1 = sys.argv[1], datetime.fromisoformat(sys.argv[2]), datetime.fromisoformat(sys.argv[3])
    insts = sys.argv[4:] or ["prefill-0", "prefill-1"]
    W = (t1 - t0).total_seconds()
    ranks = {}
    for inst in insts:
        rows = defaultdict(list)
        for line in open(f"{logs}/{inst}.log", errors="ignore"):
            m = PAT.match(line)
            if not m:
                continue
            t = datetime.fromisoformat(m.group(1)[:26].rstrip("Z"))
            if t0 <= t <= t1:
                rows[m.group(2)].append(((t - t0).total_seconds(), int(m.group(3)), int(m.group(4)),
                                         int(m.group(5)), int(m.group(6)), int(m.group(7))))
        for dp, r in sorted(rows.items()):
            ranks[f"{inst}/DP{dp}"] = r
    print(f"window {sys.argv[2]} .. {sys.argv[3]}")
    tsum = tfull = 0.0
    nsum = nfull = 0
    for name, r in ranks.items():
        new = sum(x[2] for x in r)
        busy = backlog = 0.0
        for a, b in zip(r, r[1:]):
            dt = b[0] - a[0]
            if dt < 3.0:
                busy += dt
                tsum += dt
                nsum += a[2]
                if a[1] == 1 and a[2] >= 8000:
                    tfull += dt
                    nfull += a[2]
            if a[4] > 0 or a[5] > 0:
                backlog += dt
        q = sorted(x[4] for x in r)
        runs, cur = [], 0
        for x in r:
            if x[1] == 1 and x[2] >= 8000:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        print(f"{name}: batches {len(r):5d} new {new / 1e6:6.1f}M ({new / W:5.0f} tok/s) busy {busy / W:4.0%} "
              f"backlogged {backlog / W:4.0%} queue mean {sum(q) / max(1, len(q)):4.1f} p90 {q[int(.9 * len(q))] if q else 0:3d} "
              f"long runs(>=5 chunks) {sum(1 for x in runs if x >= 5):3d} longest {max(runs) if runs else 0:3d}")
    print(f"full single-request chunks: {tfull / max(tsum, 1e-9):.0%} of prefill busy time, {nfull / max(nsum, 1):.0%} of new tokens")
    names = list(ranks)
    times = {n: [x[0] for x in ranks[n]] for n in names}

    def state(n, s):
        i = bisect.bisect_right(times[n], s) - 1
        if i < 0 or s - ranks[n][i][0] >= 3.0:
            return False, 0
        return True, ranks[n][i][4]

    idle_while_backlog, spread = 0, []
    for s in range(int(W)):
        st = [state(n, s) for n in names]
        if any(not b for b, _ in st) and any(q >= 5 for _, q in st):
            idle_while_backlog += 1
        qs = [q for _, q in st]
        spread.append(max(qs) - min(qs))
    spread.sort()
    print(f"seconds with a rank idle while another has queue >= 5: {idle_while_backlog / W:.0%}; "
          f"queue spread across ranks median {spread[len(spread) // 2]} p90 {spread[int(.9 * len(spread))]}")


if __name__ == "__main__":
    main()
