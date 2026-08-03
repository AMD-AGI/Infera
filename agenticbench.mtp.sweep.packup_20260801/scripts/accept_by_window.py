#!/usr/bin/env python3
"""Per-point MTP acceptance, binned from the decode leg's own timestamped log.

WHY NOT THE /server_info SNAPSHOT. `avg_spec_accept_length` is

    metrics_reporter.spec_total_num_accept_tokens
    / metrics_reporter.spec_total_num_forward_ct          (scheduler.py:3787)

i.e. a CUMULATIVE mean over the engine's whole lifetime, and it is reported
PER DP RANK (8 entries in `internal_states`, one per rank, and they differ:
1.42-1.58 at the last point). Snapshotting it after each sweep point therefore
gives neither a per-point value nor a fleet value -- the apparent monotone
decline across the sweep is mostly dilution by earlier traffic. Differencing
consecutive snapshots would need the underlying counters, which /server_info
does not expose.

The decode log instead stamps `accept len: X` on every decode batch, per rank,
with a timestamp -- 12,654 samples across this sweep. Binning those into each
point's [start, end) window gives a real per-point distribution across all 8
ranks.

Window boundaries come from the sweep driver's own `##### point ... HH:MM:SS`
banners, so they are the measured windows, not reconstructed ones.

Usage: accept_by_window.py <decode.log> <sweep.log> [out.md]
"""
import re
import sys
from datetime import datetime

DEC, SWP = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else None

# ---- window boundaries from the sweep banners -----------------------------
wins = []
banner = re.compile(r"^##### (p\d+)\s+ISL=(\d+) OSL=(\d+) conc=(\d+).*?(\d\d:\d\d:\d\d)\s*$")
done = re.compile(r"^############ done (\S+)")
end_utc = None
with open(SWP, errors="ignore") as f:
    for line in f:
        m = banner.match(line.strip())
        if m:
            wins.append([m.group(1), int(m.group(4)), m.group(5), None])
        d = done.match(line.strip())
        if d:
            end_utc = d.group(1)[11:19]
for i in range(len(wins) - 1):
    wins[i][3] = wins[i + 1][2]
if wins and end_utc:
    wins[-1][3] = end_utc

# ---- accept-len samples ---------------------------------------------------
samp = re.compile(r"^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) DP(\d+).*accept len: ([0-9.]+)")
rows = []
with open(DEC, errors="ignore", encoding="utf-8", newline="") as f:
    for line in f:
        m = samp.match(line.strip())
        if m:
            rows.append((m.group(2), int(m.group(3)), float(m.group(4))))


def pct(a, p):
    if not a:
        return None
    a = sorted(a)
    k = (len(a) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(a) - 1)
    return a[lo] if lo == hi else a[lo] + (a[hi] - a[lo]) * (k - lo)


L = []
A = L.append
A("### MTP acceptance per sweep point — binned from the decode leg's log\n")
A(f"Source: {len(rows):,} timestamped `accept len` samples across all 8 DP ranks.\n")
A("| point | conc | window (UTC) | n | mean | p10 | p50 | p90 | max | ranks |")
A("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
for pt, conc, t0, t1 in wins:
    if not t1:
        continue
    vals = [v for (t, r, v) in rows if t0 <= t < t1]
    ranks = len({r for (t, r, v) in rows if t0 <= t < t1})
    if not vals:
        A(f"| {pt} | {conc} | {t0}–{t1} | 0 | - | - | - | - | - | - |")
        continue
    A(f"| {pt} | {conc} | {t0}–{t1} | {len(vals):,} | {sum(vals)/len(vals):.3f} | "
      f"{pct(vals,10):.2f} | {pct(vals,50):.2f} | {pct(vals,90):.2f} | {max(vals):.2f} | {ranks} |")

txt = "\n".join(L)
print(txt)
if OUT:
    with open(OUT, "w") as f:
        f.write(txt + "\n")
