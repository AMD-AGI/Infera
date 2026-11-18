#!/usr/bin/env python3
"""Per-call view of selected kernels in one torch-profiler trace: duration distribution and launch grids.
  trace_kernels.py TRACE REGEX [REGEX ...]
For each kernel name matching a regex: calls, total, p10/p50/p90 duration, and the most common (grid, block) launch
shapes with their mean duration.
"""
import collections
import gzip
import json
import re
import sys


def main():
    path, pats = sys.argv[1], [re.compile(p) for p in sys.argv[2:]]
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        ev = json.load(f)
    ev = ev.get("traceEvents", ev)
    by = collections.defaultdict(list)
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        n = e.get("name", "")
        if any(p.search(n) for p in pats):
            a = e.get("args", {})
            by[n].append((e.get("dur", 0), str(a.get("grid")), str(a.get("block"))))
    for n, v in sorted(by.items(), key=lambda kv: -sum(x[0] for x in kv[1])):
        d = sorted(x[0] for x in v)
        q = lambda p: d[min(len(d) - 1, int(p * len(d)))]
        print(f"{n[:110]}\n  calls {len(d)} total {sum(d) / 1e3:.1f} ms  p10 {q(0.1):.1f} p50 {q(0.5):.1f} p90 {q(0.9):.1f} us")
        shapes = collections.defaultdict(list)
        for dur, g, b in v:
            shapes[(g, b)].append(dur)
        for (g, b), ds in sorted(shapes.items(), key=lambda kv: -len(kv[1]))[:4]:
            print(f"    grid {g} block {b}: {len(ds)} calls, mean {sum(ds) / len(ds):.1f} us")


if __name__ == "__main__":
    main()
