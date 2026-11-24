#!/usr/bin/env python3
"""Mean queue state of a vLLM engine from its API-server stream log (plain or .gz), summed over DP engines per 10 s line.
  engine_queue.py LOG HH:MM:SS HH:MM:SS [LOG HH:MM:SS HH:MM:SS ...]
For comparing runs mid-flight on the same measurement minutes (e.g. a prefill change against its same-node baseline):
prompt / generation tok/s, Running, Waiting, Deferred (waiting for a CPU-tier load), KV fetch waiting to start /
in progress / completed waiting, GPU KV usage (mean over engines). Deferred and KV fetch fields are only logged
when non-zero, so a missing field counts as 0. By Little's law, mean count / request rate = mean time in that state.
Standard library only (runs on the login node).
"""
import gzip
import re
import sys
from collections import defaultdict

LINE = re.compile(r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) \[loggers.py:\d+\] Engine (\d+): (.*)")
FIELDS = {
    "prompt": re.compile(r"Avg prompt throughput: ([0-9.]+)"),
    "gen": re.compile(r"Avg generation throughput: ([0-9.]+)"),
    "running": re.compile(r"Running: (\d+) reqs"),
    "waiting": re.compile(r"Waiting: (\d+) reqs"),
    "deferred": re.compile(r"Deferred: (\d+) reqs"),
    "fetch_wait": re.compile(r"(\d+) waiting to start"),
    "fetch_prog": re.compile(r"(\d+) in progress"),
    "fetch_done": re.compile(r"(\d+) completed waiting"),
    "kv": re.compile(r"GPU KV cache usage: ([0-9.]+)%"),
}
SUMMED = ["prompt", "gen", "running", "waiting", "deferred", "fetch_wait", "fetch_prog", "fetch_done"]


def secs(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def one(path, t0, t1):
    a, b = secs(t0), secs(t1)
    op = gzip.open if path.endswith(".gz") else open
    rows = defaultdict(lambda: defaultdict(float))
    engines = defaultdict(set)
    for line in op(path, "rt", errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        t = secs(m.group(1))
        if b < a:
            t = t if t >= a else t + 86400
            inside = a <= t <= b + 86400
        else:
            inside = a <= t <= b
        if not inside:
            continue
        # engines log a few ms apart: bucket on the 10 s line
        key = t // 10
        body = m.group(3)
        engines[key].add(m.group(2))
        for f, rx in FIELDS.items():
            mm = rx.search(body)
            rows[key][f] += float(mm.group(1)) if mm else 0.0
    if not rows:
        print(f"{path}: no logger lines in {t0}..{t1}")
        return
    n = len(rows)
    mean = {f: sum(r[f] for r in rows.values()) / n for f in FIELDS}
    neng = sum(len(e) for e in engines.values()) / n
    mean["kv"] /= max(neng, 1)
    print(f"{path} {t0}..{t1}: {n} lines, {neng:.1f} engines per line")
    print("  " + ", ".join(f"{f} {mean[f]:.2f}" if f not in ("prompt", "gen") else f"{f} {mean[f]:,.0f} tok/s"
                          for f in SUMMED) + f", KV usage {mean['kv']:.1f}%")


def main():
    args = sys.argv[1:]
    if not args or len(args) % 3:
        sys.exit(__doc__)
    for i in range(0, len(args), 3):
        one(*args[i:i + 3])


if __name__ == "__main__":
    main()
