#!/usr/bin/env python3
"""Mooncake transfer worker time split in round 2 from KVWDIAG lines (diag/conn.py) of a gspbw run, per prefill PP rank
and worker thread: chunks, idle (blocked in queue.get), busy, and inside busy the waits on wait_event / indices_event.
Usage: kvwdiag.py NAME (logs/gsp/NAME)"""
import re, sys
from pathlib import Path
name = sys.argv[1]
g, r = Path('logs/gsp') / name, Path('logs/runs') / name
t0, t1 = float((g / 'r2_start').read_text()), float((g / 'r2_end').read_text())
pat = re.compile(r'(PP\d)[^\]]*\].*KVWDIAG t=([\d.]+) worker=(.+?) n=(\d+) idle=([\d.]+) busy=([\d.]+) wait_event=([\d.]+) indices_event=([\d.]+)')
by = {}
for line in (r / 'logs/glm52_agentx-prefill0.log').read_text(errors='replace').splitlines():
    m = pat.search(line)
    if m:
        by.setdefault((m.group(1), m.group(3)), []).append(tuple(float(x) for x in (m.group(2),) + m.groups()[3:]))
print(f"{name}: round 2 = {t1 - t0:.0f} s")
for k in sorted(by):
    rows = by[k]
    a = max((x for x in rows if x[0] <= t0), default=(t0, 0, 0, 0, 0, 0))
    b = max((x for x in rows if x[0] <= t1 + 5), default=a)
    d = [y - x for x, y in zip(a, b)]
    if d[1] <= 0:
        continue
    print(f"  {k[0]} {k[1]}: chunks {d[1]:.0f}, idle {d[2]:.0f} s, busy {d[3]:.0f} s (wait_event {d[4]:.0f} s, indices_event {d[5]:.0f} s), "
          f"busy/chunk {d[3] / d[1] * 1000:.0f} ms")
