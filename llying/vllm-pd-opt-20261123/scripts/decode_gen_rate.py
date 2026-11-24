#!/usr/bin/env python3
"""Decode generation rate from a vLLM API-server stream log (plain or .gz), summed over the DP engines per 10 s line.
  decode_gen_rate.py DECODE_LOG HH:MM:SS HH:MM:SS
Prints the mean generation tok/s over the window and the mean running / waiting sums, for comparing runs mid-flight on
the same measurement minutes.
"""
import gzip
import re
import sys

PAT = re.compile(r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) \[loggers.py:\d+\] Engine (\d+): Avg prompt throughput: [0-9.]+ tokens/s, "
                 r"Avg generation throughput: ([0-9.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+)")


def main():
    path, t0, t1 = sys.argv[1], sys.argv[2], sys.argv[3]
    op = gzip.open if path.endswith(".gz") else open
    gen, run, wait = {}, {}, {}
    for line in op(path, "rt", errors="replace"):
        m = PAT.search(line)
        if not m:
            continue
        t, _, g, r, w = m.groups()
        if not t0 <= t <= t1:
            continue
        k = t[:7]
        gen[k] = gen.get(k, 0.0) + float(g)
        run[k] = run.get(k, 0) + int(r)
        wait[k] = wait.get(k, 0) + int(w)
    n = max(1, len(gen))
    print(f"{path} {t0}-{t1}: {len(gen)} buckets; generation tok/s (sum of engines) {sum(gen.values()) / n:.0f}; "
          f"running {sum(run.values()) / n:.1f}; waiting {sum(wait.values()) / n:.1f}")


if __name__ == "__main__":
    main()
