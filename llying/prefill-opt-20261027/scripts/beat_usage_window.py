"""PP-rank "Prefill batch" lines of a prefill log as rows, with per-window beat / usage summaries, and optionally the
rows of one window listed one per batch (gap to the next batch of the same rank, batch composition, usage, queue).
  python3 beat_usage_window.py PREFILL_LOG [--rank 0] [--win HH:MM:SS-HH:MM:SS ...] [--list HH:MM:SS-HH:MM:SS]"""

import argparse
import re
import statistics as st
from datetime import datetime

PAT = re.compile(r"^(\S+)Z \[\S+ (\d\d:\d\d:\d\d) PP(\d)\] Prefill batch, #new-seq: (\d+), #new-token: (\d+), "
                 r"#cached-token: (\d+), token usage: ([\d.]+), #running-req: \d+, #queue-req: (\d+), "
                 r"#pending-token: (\d+), #bootstrap-req: (\d+), #inflight-req: (\d+)")


def load(path, rank=None):
    rows = []
    for line in open(path, errors="replace"):
        m = PAT.search(line)
        if not m or (rank is not None and int(m.group(3)) != rank):
            continue
        rows.append(dict(t=datetime.fromisoformat(m.group(1)[:26]).timestamp(), hms=m.group(1)[11:23], rank=int(m.group(3)),
                         seqs=int(m.group(4)), new=int(m.group(5)), cached=int(m.group(6)), usage=float(m.group(7)),
                         queue=int(m.group(8)), pending=int(m.group(9)), bootstrap=int(m.group(10)),
                         inflight=int(m.group(11))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--win", nargs="*", default=[])
    ap.add_argument("--list")
    a = ap.parse_args()
    rows = load(a.log, a.rank)
    for i in range(len(rows) - 1):
        rows[i]["gap"] = rows[i + 1]["t"] - rows[i]["t"]
    rows[-1]["gap"] = None
    for w in a.win:
        lo, hi = w.split("-")
        r = [x for x in rows if lo <= x["hms"][:8] <= hi]
        g = [x["gap"] for x in r if x["gap"] is not None and x["queue"] > 0]
        u = [x["usage"] for x in r]
        if not r:
            print(f"{w}: no rows")
            continue
        print(f"{w}: {len(r)} batches, queued beats {len(g)} p50 {st.median(g) * 1e3:.0f} mean {st.mean(g) * 1e3:.0f} ms; "
              f"usage min / p50 / max {min(u):.2f} / {st.median(u):.2f} / {max(u):.2f}")
    if a.list:
        lo, hi = a.list.split("-")
        r = [x for x in rows if lo <= x["hms"][:8] <= hi]
        for i, x in enumerate(r):
            gap = f"{x['gap'] * 1e3:6.0f}" if x["gap"] is not None else "     -"
            print(f"{i:3d} {x['hms']} gap_next {gap} seq {x['seqs']:2d} new {x['new']:5d} cached {x['cached']:7d} "
                  f"usage {x['usage']:.2f} queue {x['queue']:2d} inflight {x['inflight']:2d} boot {x['bootstrap']:2d}")
