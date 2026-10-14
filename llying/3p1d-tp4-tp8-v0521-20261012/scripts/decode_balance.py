#!/usr/bin/env python3
"""Per-DP-rank decode load inside one AgentX profiling window, from the decode server log.

Usage: decode_balance.py DECODE_LOG START_ISO END_ISO
Reads "Decode batch" lines (one per DP rank per log interval) and prints, per rank, the
mean running requests, mean KV token usage, share of lines with usage > 0.90, mean
pre-allocation queue, mean transfer queue and mean accept length; then the same pooled
over ranks and the hottest-vs-coldest usage spread.
"""

import re
import statistics
import sys

LINE = re.compile(
    r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S* \[[^\]]*DP(\d+) TP\d+\] Decode batch, #running-req: (\d+), "
    r"#token: \d+, token usage: ([0-9.]+), accept len: ([0-9.]+).*?#prealloc-req: (\d+), #transfer-req: (\d+)")


def main() -> int:
    path, start, end = sys.argv[1:4]
    ranks = {}
    with open(path, errors="replace") as stream:
        for line in stream:
            m = LINE.match(line)
            if not m or not (start <= m.group(1) <= end):
                continue
            dp = int(m.group(2))
            ranks.setdefault(dp, []).append(tuple(float(x) for x in m.groups()[2:]))
    if not ranks:
        print("no Decode batch lines in window")
        return 1
    print(f"window {start} .. {end}")
    print("dp  lines  running  usage  >0.90  prealloc  transfer  accept")
    means = {}
    for dp in sorted(ranks):
        rows = ranks[dp]
        col = list(zip(*rows))
        means[dp] = statistics.mean(col[1])
        print(f"{dp:<3} {len(rows):>5}  {statistics.mean(col[0]):7.2f}  {means[dp]:5.2f}  "
              f"{sum(u > 0.90 for u in col[1]) / len(rows):5.1%}  {statistics.mean(col[3]):8.2f}  "
              f"{statistics.mean(col[4]):8.2f}  {statistics.mean(col[2]):6.2f}")
    allrows = [r for rows in ranks.values() for r in rows]
    col = list(zip(*allrows))
    print(f"all {len(allrows):>5}  {statistics.mean(col[0]):7.2f}  {statistics.mean(col[1]):5.2f}  "
          f"{sum(u > 0.90 for u in col[1]) / len(allrows):5.1%}  {statistics.mean(col[3]):8.2f}  "
          f"{statistics.mean(col[4]):8.2f}  {statistics.mean(col[2]):6.2f}")
    print(f"rank mean-usage spread: max {max(means.values()):.2f} / min {min(means.values()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
