#!/usr/bin/env python3
"""Router-independent engine pace inside a window, to tell stack drift from routing effects.

Usage: engine_pace.py SERVER_LOG_DIR DECODE_LOG_NAME START_ISO END_ISO
Decode: ms per decode step (log interval = 40 steps, DP ranks in lockstep) binned by the
BUSIEST rank's #running-req in the same 3 s bin, since the global step follows that rank.
Prefill: ms between consecutive batches of one rank while its queue is non-empty and the
earlier batch was a full per-rank chunk (8192 new tokens), i.e. the time of a full chunk.
"""

import collections
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

DECODE = re.compile(r"^(\S+)Z \[[^\]]*DP(\d+) TP\d+\] Decode batch, #running-req: (\d+)")
PREFILL = re.compile(
    r"^(\S+)Z \[[^\]]*DP(\d+) TP\d+\] Prefill batch, #new-seq: \d+, #new-token: (\d+), "
    r"#cached-token: \d+, token usage: [0-9.]+, #running-req: \d+, #queue-req: (\d+)")
DECODE_LOG_INTERVAL = 40
FULL_CHUNK = 8192


def ts(text: str) -> datetime:
    return datetime.fromisoformat(text[:26])


def decode_pace(path: Path, start: datetime, end: datetime) -> None:
    rows = []
    with open(path, errors="replace") as stream:
        for line in stream:
            m = DECODE.match(line)
            if m:
                t = ts(m.group(1))
                if start <= t <= end:
                    rows.append((t, int(m.group(2)), int(m.group(3))))
    last, steps = {}, []
    for t, dp, _ in rows:
        if dp in last:
            ms = (t - last[dp]).total_seconds() * 1000 / DECODE_LOG_INTERVAL
            if ms < 500:
                steps.append((t, ms))
        last[dp] = t
    bins = collections.defaultdict(dict)
    for t, dp, run in rows:
        k = int(t.timestamp() // 3)
        bins[k][dp] = max(bins[k].get(dp, 0), run)
    busiest = {k: max(v.values()) for k, v in bins.items() if len(v) >= 6}
    by = collections.defaultdict(list)
    for t, ms in steps:
        k = int(t.timestamp() // 3)
        if k in busiest:
            by[busiest[k]].append(ms)
    cols = list(range(4, 15))
    print("decode ms/step by busiest-rank #running-req [samples]")
    print("".join(f"{c:>12}" for c in cols))
    print("".join(f"{statistics.median(by[c]):7.1f}[{len(by[c]):>3}]" if len(by[c]) >= 30
                  else f"{'-':>12}" for c in cols))


def prefill_pace(logs: Path, start: datetime, end: datetime) -> None:
    print(f"prefill ms per full {FULL_CHUNK}-token chunk while queued (median / p90 [samples])")
    for log in sorted(logs.glob("prefill-[0-9].log")):
        prev, samples = {}, []
        with open(log, errors="replace") as stream:
            for line in stream:
                m = PREFILL.match(line)
                if not m:
                    continue
                t = ts(m.group(1))
                if not start <= t <= end:
                    continue
                dp, new, queue = int(m.group(2)), int(m.group(3)), int(m.group(4))
                if dp in prev:
                    pt, pnew, pqueue = prev[dp]
                    if pnew >= FULL_CHUNK and pqueue > 0:
                        samples.append((t - pt).total_seconds() * 1000)
                prev[dp] = (t, new, queue)
        if len(samples) >= 30:
            samples.sort()
            print(f"  {log.stem:10} {statistics.median(samples):7.0f} / "
                  f"{samples[int(len(samples) * 0.9)]:7.0f} [{len(samples)}]")
        else:
            print(f"  {log.stem:10} - [{len(samples)}]")


def main() -> int:
    logs, decode_log, start, end = Path(sys.argv[1]), sys.argv[2], ts(sys.argv[3]), ts(sys.argv[4])
    print(f"window {sys.argv[3]} .. {sys.argv[4]}")
    decode_pace(logs / decode_log, start, end)
    prefill_pace(logs, start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
