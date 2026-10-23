"""Prefill time series from an engine log, one line per minute (every STEP-th minute): the fullest rank's KV usage,
and waiting and in-flight requests summed over ranks. A PP prefill counts stage 0 only.
  python3 prefill_series.py PREFILL_LOG|- HH:MM HH:MM STEP"""
import collections
import re
import sys

pat = re.compile(r"^(\S+)Z \[\S+ \S+ (?:DP(\d) TP\d|PP(\d))\] Prefill batch.*token usage: ([0-9.]+).*#queue-req: (\d+)"
                 r".*#inflight-req: (\d+)")
b = collections.defaultdict(dict)
for line in open(sys.argv[1], errors="replace") if sys.argv[1] != "-" else sys.stdin:
    m = pat.search(line)
    if not m or m.group(3) not in (None, "0"):
        continue
    b[m.group(1)[11:16]][int(m.group(2) or 0)] = (float(m.group(4)), int(m.group(5)), int(m.group(6)))
lo, hi = sys.argv[2], sys.argv[3]
for t in sorted(b)[:: int(sys.argv[4])]:
    if lo <= t <= hi:
        v = b[t].values()
        print(t, f"usage max {max(x[0] for x in v):.2f}", "queue", sum(x[1] for x in v), "inflight", sum(x[2] for x in v))
