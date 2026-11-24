#!/usr/bin/env python3
"""Stalls of an engine's busy loop, from its API-server stream log (plain or .gz): gaps between consecutive 10 s logger
lines of one engine, and stretches where prompt (prefill) or generation (decode) throughput is ~0 while requests wait.
  engine_stalls.py LOG HH:MM:SS SECONDS [ENGINE] [GAP_S]
(start of the measurement and its length, e.g. the "measurement" line of ttft_breakdown.txt; ENGINE default 000;
GAP_S default 15). Prints each gap > GAP_S and each idle-while-waiting stretch >= 20 s, and their totals.
Standard library only.
"""
import gzip
import re
import sys

LINE = re.compile(r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) \[loggers.py:\d+\] Engine (\d+): Avg prompt throughput: ([0-9.]+) "
                  r"tokens/s, Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+)")


def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def hms(t):
    t %= 86400
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


def main():
    path, t0, dur = sys.argv[1], secs(sys.argv[2]), int(sys.argv[3])
    eng = sys.argv[4] if len(sys.argv) > 4 else "000"
    gap_s = float(sys.argv[5]) if len(sys.argv) > 5 else 15
    op = gzip.open if path.endswith(".gz") else open
    rows, last = [], None
    for line in op(path, "rt", errors="replace"):
        m = LINE.search(line)
        if not m or m.group(2) != eng:
            continue
        t = secs(m.group(1))
        if last is not None and t < last - 43200:
            t += 86400
        last = t
        if t0 <= t <= t0 + dur:
            rows.append((t, float(m.group(3)), float(m.group(4)), int(m.group(5)), int(m.group(6))))
    gaps = [(a[0], b[0] - a[0]) for a, b in zip(rows, rows[1:]) if b[0] - a[0] > gap_s]
    idle, start = [], None
    for t, p, g, run, wait in rows:
        busy = p + g > 500
        if not busy and wait > 0:
            start = t if start is None else start
        else:
            if start is not None and t - start >= 20:
                idle.append((start, t - start))
            start = None
    print(f"{path} engine {eng}, {hms(t0)} + {dur} s: {len(rows)} lines")
    print(f"  logger gaps > {gap_s:g} s: {len(gaps)}, total {sum(d for _, d in gaps)} s: "
          + " ".join(f"{hms(t)}+{d}s" for t, d in gaps))
    print(f"  ~0 throughput while waiting >= 20 s: {len(idle)}, total {sum(d for _, d in idle)} s: "
          + " ".join(f"{hms(t)}+{d}s" for t, d in idle))


if __name__ == "__main__":
    main()
