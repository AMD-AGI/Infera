#!/usr/bin/env python3
"""MTP acceptance from decode logs three ways: plain mean over 'Decode batch' lines (what the gate prints), weighted by
#running-req, and over lines with >= 8 running requests only. Usage: accept_weighted.py RUN_DIR ..."""
import re
import sys

R = re.compile(r"#running-req: (\d+),.*accept len: ([0-9.]+)")
for d in sys.argv[1:]:
    n = s = w = sw = s8 = n8 = 0
    for line in open(f"{d}/logs/glm52_2p1d-decode.log", errors="ignore"):
        m = R.search(line)
        if not m:
            continue
        run, acc = int(m.group(1)), float(m.group(2))
        n += 1; s += acc; w += run; sw += acc * run
        if run >= 8:
            n8 += 1; s8 += acc
    print(f"{d.rsplit('/', 1)[-1]:48s} lines {n:5d}  plain {s / n:.3f}  req-weighted {sw / w:.3f}  "
          f"running>=8 {s8 / max(n8, 1):.3f} ({n8} lines)")
