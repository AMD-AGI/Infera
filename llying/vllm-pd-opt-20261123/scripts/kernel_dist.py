#!/usr/bin/env python3
"""Duration distribution of the kernels whose name matches each pattern in a torch trace (.json / .json.gz).
  kernel_dist.py TRACE PATTERN [PATTERN ...]
For a collective the low percentiles approximate the transfer itself (this rank arrived last) and the rest is waiting
for slower ranks.
"""
import gzip
import json
import re
import sys


def main():
    path, pats = sys.argv[1], [re.compile(p) for p in sys.argv[2:]]
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        ev = json.load(f)["traceEvents"]
    for p in pats:
        d = sorted(e["dur"] for e in ev if e.get("ph") == "X" and e.get("cat") == "kernel" and p.search(e.get("name", "")))
        if not d:
            print(f"{p.pattern}: none")
            continue
        q = lambda x: d[min(len(d) - 1, int(x * len(d)))]
        print(f"{p.pattern}: n {len(d)}; us min {d[0]:.1f} p5 {q(0.05):.1f} p10 {q(0.1):.1f} p25 {q(0.25):.1f} "
              f"p50 {q(0.5):.1f} p75 {q(0.75):.1f} p90 {q(0.9):.1f} mean {sum(d) / len(d):.1f}")


if __name__ == "__main__":
    main()
