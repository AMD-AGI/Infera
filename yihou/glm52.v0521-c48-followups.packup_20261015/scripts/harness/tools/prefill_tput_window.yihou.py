#!/usr/bin/env python3
"""DP0 prefill per-batch input-throughput distribution in a window after AgentX profiling start.
Usage: prefill_tput_window.yihou.py MINUTES LABEL:PREFILL_LOG:BENCHMARK_LOG:DATE [...]
(benchmark.log carries HH:MM:SS only; DATE=YYYY-MM-DD of the run, both clocks UTC)."""
import re, subprocess, sys, statistics as st
from datetime import datetime, timedelta
win = float(sys.argv[1])
pat = re.compile(r'^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.\d+Z \[[^\]]*DP0 TP0\] Prefill batch.*?#new-token: (\d+), #cached-token: (\d+).*input throughput \(token/s\): ([\d.]+)')
def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
print(f"window = first {win:g} min of the profiling phase; DP0 TP0 'Prefill batch' lines")
print(f"{'run':8} {'start':19} {'n':>5} {'per_min':>7} {'mean':>8} {'p10':>8} {'p50':>8} {'p90':>8} {'<1000':>6} {'new_tok_sum':>12} {'cached_sum':>12}")
for spec in sys.argv[2:]:
    label, plog, blog, date = spec.split(':')
    s = subprocess.run(['strings', blog], capture_output=True, text=True).stdout
    hhmmss = re.search(r'^(\d\d:\d\d:\d\d)\.\d+ INFO\s+Credit phase start: profiling', s, re.M).group(1)
    t0 = datetime.fromisoformat(f"{date}T{hhmmss}"); t1 = t0 + timedelta(minutes=win)
    tp, nt, ct = [], 0, 0
    for line in subprocess.run(['strings', plog], capture_output=True, text=True).stdout.splitlines():
        m = pat.match(line)
        if m and t0 <= datetime.fromisoformat(m.group(1)) < t1:
            tp.append(float(m.group(4))); nt += int(m.group(2)); ct += int(m.group(3))
    print(f"{label:8} {t0.isoformat():19} {len(tp):5d} {len(tp)/win:7.1f} {st.mean(tp):8.0f} {q(tp,.1):8.0f} {q(tp,.5):8.0f} {q(tp,.9):8.0f} {sum(x<1000 for x in tp):6d} {nt:12d} {ct:12d}")
