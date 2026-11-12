#!/usr/bin/env python3
"""KV transfer contiguity in round 2 from the diagnostic KVDIAG lines (diag/utils.py) of a gspbd run: per prefill PP rank,
indices (pages) per transfer block = how fragmented the prefill page lists were. Usage: kvdiag.py NAME (logs/gsp/NAME)"""
import re, sys
from pathlib import Path
name = sys.argv[1]
g, r = Path('logs/gsp') / name, Path('logs/runs') / name
t0, t1 = float((g / 'r2_start').read_text()), float((g / 'r2_end').read_text())
pat = re.compile(r'(PP\d)[^\]]*\].*KVDIAG t=([\d.]+) calls=(\d+) idx=(\d+) blocks=(\d+) src_brk=(\d+) dst_brk=(\d+)')
by = {}
for line in (r / 'logs/glm52_agentx-prefill0.log').read_text(errors='replace').splitlines():
    m = pat.search(line)
    if m:
        by.setdefault(m.group(1), []).append(tuple(float(x) for x in m.groups()[1:]))
for pp in sorted(by):
    rows = by[pp]
    a = max((x for x in rows if x[0] <= t0), default=(t0, 0, 0, 0, 0, 0))
    b = max((x for x in rows if x[0] <= t1), default=a)
    d = [y - x for x, y in zip(a, b)]
    print(f"{name} {pp}: r2 calls {d[1]:.0f}, pages {d[2]:.0f}, blocks {d[3]:.0f}, pages/block {d[2] / max(d[3], 1):.1f}, "
          f"src breaks {d[4]:.0f}, dst breaks {d[5]:.0f}")
