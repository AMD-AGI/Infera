#!/usr/bin/env python3
"""Mooncake transfer stats of a prefill log over a time window, from its 10 s "MooncakeConnector={...}" metric lines
(weighted by the number of transfers in each line).
  mooncake_window.py LOG HH:MM:SS HH:MM:SS [LOG HH:MM:SS HH:MM:SS ...]
"""
import re
import sys

pat = re.compile(r"INFO \d\d-\d\d (\d\d:\d\d:\d\d) .*MooncakeConnector=\{'Num successful transfers': (\d+), "
                 r"'Avg xfer time \(ms\)': ([0-9.]+), 'P90 xfer time \(ms\)': ([0-9.]+), 'Avg MB per transfer': ([0-9.]+), "
                 r"'Throughput \(MB/s\)': ([0-9.]+), 'Avg number of descriptors': ([0-9.]+)")


def s(t):
    h, m, x = map(int, t.split(":"))
    return h * 3600 + m * 60 + x


for i in range(1, len(sys.argv), 3):
    path, a, b = sys.argv[i], s(sys.argv[i + 1]), s(sys.argv[i + 2])
    n = t = mb = d = p90 = 0
    for line in open(path, errors="replace"):
        m = pat.search(line)
        if not m or not a <= s(m.group(1)) < b:
            continue
        k = int(m.group(2))
        n += k; t += k * float(m.group(3)); p90 += k * float(m.group(4)); mb += k * float(m.group(5)); d += k * float(m.group(7))
    print(f"{path} {sys.argv[i + 1]}-{sys.argv[i + 2]}: transfers {n}, avg {t / n:.1f} ms, p90 (avg of 10 s) {p90 / n:.0f} ms, "
          f"{mb / n / 1000:.2f} GB, {d / n:.0f} desc, {mb / t:.2f} GB/s per transfer")
