#!/usr/bin/env python3
"""stepbins.py LABEL LOG START END [LABEL LOG START END ...]: decode step time vs running batch, TP0 lines.

step_ms = 1000 * running * accept_len / gen_throughput (gen throughput is per decode-log interval, so this is
wall time per decode step over that interval). "clean" = no Prefill batch line since the previous decode line."""
import re, sys
HEAD = re.compile(r"\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d) TP0[^\]]*\] (Decode|Prefill) batch")
DEC = re.compile(r"#running-req: (\d+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)")
BINS = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 40), (41, 80)]

def rows(log, start, end):
    pf = False
    for line in open(log, errors="replace"):
        m = HEAD.search(line)
        if not m or not (start <= m.group(1) <= end):
            continue
        if m.group(2) == "Prefill":
            pf = True
            continue
        d = DEC.search(line)
        run, acc, gen = int(d.group(1)), float(d.group(2)), float(d.group(3))
        if gen > 0:
            yield run, 1000.0 * run * acc / gen, not pf
        pf = False

args = sys.argv[1:]
print("label | bin | n_all | step_ms all (median/mean) | n_clean | step_ms clean (median/mean)")
for i in range(0, len(args), 4):
    label, log, start, end = args[i:i + 4]
    r = list(rows(log, start, end))
    for lo, hi in BINS:
        a = sorted(s for run, s, c in r if lo <= run <= hi)
        c = sorted(s for run, s, cl in r if lo <= run <= hi and cl)
        f = lambda v: "%.1f / %.1f" % (v[len(v) // 2], sum(v) / len(v)) if v else "-"
        print(f"{label} | {lo}-{hi} | {len(a)} | {f(a)} | {len(c)} | {f(c)}")
