#!/usr/bin/env python3
"""bneckw (llr7r analysis 2026-10-24): bneck.yihou.py with an explicit window.

usage: python3 -I bneckw.yihou.py <run dir with logs/> <start 'YYYY-mm-dd HH:MM:SS'> <end ...> [exclude_start exclude_end]
Same statistics as bneck.yihou.py; optional exclusion sub-window (e.g. a profiling perturbation).
"""
import re, sys, statistics as st
from datetime import datetime

run = sys.argv[1]
P = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S')
t0, t1 = P(sys.argv[2]), P(sys.argv[3])
ex = (P(sys.argv[4]), P(sys.argv[5])) if len(sys.argv) > 5 else None
TS = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\w+)(?: (TP\d+))?\]')
KV = re.compile(r'([#\w\- ()/]+?): ([\d.]+|True|False)(?:,|$)')


def inwin(t):
    return t0 <= t <= t1 and not (ex and ex[0] <= t <= ex[1])


def parse(path, key):
    out = []
    with open(path, errors='replace') as f:
        for line in f:
            if key not in line:
                continue
            m = TS.match(line)
            if not m:
                continue
            t = P(m.group(1))
            if not inwin(t):
                continue
            d = {k.strip(): v for k, v in KV.findall(line.split(key, 1)[1])}
            out.append((t, m.group(2), d))
    return out


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')


def f(d, k):
    try:
        return float(d[k])
    except (KeyError, ValueError):
        return float('nan')


dec = parse(f'{run}/logs/yihou-llr-decode.log', 'Decode batch,')
pre = parse(f'{run}/logs/yihou-llr-prefill0.log', 'Prefill batch,')
span_min = ((t1 - t0).total_seconds() - ((ex[1] - ex[0]).total_seconds() if ex else 0)) / 60
print(f'window {t0} .. {t1} excl {ex} ({span_min:.1f} min)')
print('\nDECODE per rank: n | running mean/p50/p90 | tokusage mean/max | prealloc-req mean/frac>0 |'
      ' transfer-req mean | queue-req mean/frac>0 | gen tok/s mean | step ms p50/p90')
for r in sorted({r for _, r, _ in dec}):
    rows = [d for t, rr, d in dec if rr == r]
    run_ = [f(d, '#running-req') for d in rows]
    tu = [f(d, 'token usage') for d in rows]
    pa = [f(d, '#prealloc-req') for d in rows]
    tr = [f(d, '#transfer-req') for d in rows]
    qu = [f(d, '#queue-req') for d in rows]
    gt = [f(d, 'gen throughput (token/s)') for d in rows]
    step = [1000 * f(d, '#running-req') * f(d, 'accept len') / f(d, 'gen throughput (token/s)')
            for d in rows if f(d, 'gen throughput (token/s)') > 1]
    print(f'{r:4} {len(rows):5} | {st.mean(run_):5.1f} {q(run_, .5):4.0f} {q(run_, .9):4.0f} |'
          f' {st.mean(tu):.2f} {max(tu):.2f} | {st.mean(pa):5.1f} {sum(x > 0 for x in pa) / len(pa):.2f} |'
          f' {st.mean(tr):5.1f} | {st.mean(qu):5.1f} {sum(x > 0 for x in qu) / len(qu):.2f} |'
          f' {st.mean(gt):7.0f} | {q(step, .5):5.1f} {q(step, .9):5.1f}')
rows = [d for t, rr, d in pre if rr == 'PP0']
if rows:
    nt = [f(d, '#new-token') for d in rows]
    ct = [f(d, '#cached-token') for d in rows]
    span = span_min * 60
    print(f'\nPREFILL PP0: batches {len(rows)} ({len(rows) / span:.2f}/s) | new tok/s {sum(nt) / span:.0f}'
          f' | cached tok/s {sum(ct) / span:.0f} | hit {sum(ct) / max(1, sum(ct) + sum(nt)):.3f}')
    for k in ('#new-seq', '#new-token', '#queue-req', '#pending-token', '#bootstrap-req', '#inflight-req',
              '#running-req', 'token usage', 'input throughput (token/s)'):
        v = [x for x in (f(d, k) for d in rows) if x == x]
        if v:
            print(f'  {k:28} mean {st.mean(v):10.2f} p50 {q(v, .5):10.2f} p90 {q(v, .9):10.2f}'
                  f' max {max(v):10.2f} frac>0 {sum(x > 0 for x in v) / len(v):.2f}')
