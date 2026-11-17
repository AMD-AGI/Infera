#!/usr/bin/env python3
"""Per-NIC deltas between two nic_snap.sh outputs: nic_diff.py A B [SECONDS]."""
import sys
a = {l.split()[0]: list(map(int, l.split()[1:])) for l in open(sys.argv[1]) if l.strip()}
b = {l.split()[0]: list(map(int, l.split()[1:])) for l in open(sys.argv[2]) if l.strip()}
secs = float(sys.argv[3]) if len(sys.argv) > 3 else 0
for n in sorted(b):
    d = [y - x for x, y in zip(a.get(n, [0] * 4), b[n])]
    rate = f" tx {d[0] / secs / 1e9:6.2f} GB/s" if secs else ""
    print(f"{n} tx {d[0] / 1e9:10.2f} GB rx {d[1] / 1e9:10.2f} GB ack_timeout {d[2]:6d} retx {d[3]:8d}{rate}")
