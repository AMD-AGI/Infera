#!/usr/bin/env python3
"""Per-prefill-instance load inside one AgentX profiling window, from the prefill logs.

Usage: prefill_balance.py SERVER_LOG_DIR START_ISO END_ISO
Sums "Prefill batch" lines per instance (prefill-*.log): batches, computed (new) tokens,
cached tokens, the cache share, and mean waiting queue / bootstrap queue per log line.
"""

import re
import statistics
import sys
from pathlib import Path

LINE = re.compile(
    r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S* \[[^\]]*DP(\d+) TP\d+\] Prefill batch, #new-seq: (\d+), "
    r"#new-token: (\d+), #cached-token: (\d+), token usage: ([0-9.]+), #running-req: \d+, "
    r"#queue-req: (\d+), #pending-token: \d+, #bootstrap-req: (\d+)")


def main() -> int:
    logs, start, end = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    print(f"window {start} .. {end}")
    print("instance   batches  new_tok(M)  cached_tok(M)  cache%  queue  bootstrap  new_tok/s")
    total_new = total_cached = 0
    seconds = None
    for log in sorted(logs.glob("prefill-[0-9].log")):
        rows = []
        with open(log, errors="replace") as stream:
            for line in stream:
                m = LINE.match(line)
                if m and start <= m.group(1) <= end:
                    rows.append(tuple(int(m.group(i)) for i in (4, 5, 7, 8)))
        if not rows:
            continue
        new = sum(r[0] for r in rows); cached = sum(r[1] for r in rows)
        total_new += new; total_cached += cached
        from datetime import datetime
        seconds = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        print(f"{log.stem:<10} {len(rows):>7}  {new / 1e6:10.1f}  {cached / 1e6:13.1f}  {cached / (new + cached):6.1%}"
              f"  {statistics.mean(r[2] for r in rows):5.2f}  {statistics.mean(r[3] for r in rows):9.2f}"
              f"  {new / seconds:9.0f}")
    if seconds:
        print(f"all        {'':>7}  {total_new / 1e6:10.1f}  {total_cached / 1e6:13.1f}"
              f"  {total_cached / (total_new + total_cached):6.1%}  {'':>5}  {'':>9}  {total_new / seconds:9.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
