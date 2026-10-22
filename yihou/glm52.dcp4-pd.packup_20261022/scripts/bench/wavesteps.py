#!/usr/bin/env python3
"""wavesteps.py LABEL LOG DECODEBENCH_JSON [...]: per wave, TP0 decode lines with running == N inside the wave window:
count, step_ms median/mean (1000*running*accept/gen_tput), accept mean, gen tput median."""
import json, re, statistics as st, sys
HEAD = re.compile(r"\[\d{4}-\d\d-\d\d (\d\d:\d\d:\d\d) TP0[^\]]*\] Decode batch")
DEC = re.compile(r"#running-req: (\d+).*?accept len: ([\d.]+).*?gen throughput \(token/s\): ([\d.]+)")
a = sys.argv[1:]
print("label | N | wave s | lines | step_ms median / mean | accept | gen tok/s median")
for i in range(0, len(a), 3):
    label, log, js = a[i:i + 3]
    lines = [(m.group(1), DEC.search(l)) for l in open(log, errors="replace") for m in [HEAD.search(l)] if m]
    for w in json.load(open(js))["waves"]:
        sel = [(int(d.group(1)), float(d.group(2)), float(d.group(3))) for t, d in lines
               if w["start"] <= t <= w["end"] and int(d.group(1)) == w["n"] and float(d.group(3)) > 0]
        if not sel:
            print(f"{label} | {w['n']} | {w['secs']} | 0 | - | - | -"); continue
        s = [1000 * r * acc / g for r, acc, g in sel]
        print(f"{label} | {w['n']} | {w['secs']} | {len(sel)} | {st.median(s):.1f} / {st.mean(s):.1f} | "
              f"{st.mean(x[1] for x in sel):.2f} | {st.median(x[2] for x in sel):.0f}")
