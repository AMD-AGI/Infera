#!/usr/bin/env python3
"""Independent verifier (llr7r logs report). usage: vlogs.yihou.py <rundir> <start> <end> <exs> <exe>"""
import sys, json, re
from datetime import datetime, timezone
P = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
rd = sys.argv[1]; t0, t1, x0, x1 = map(P, sys.argv[2:6])
inw = lambda t: t0 <= t < t1 and not (x0 <= t < x1)
ps = []; nprof = 0; nw = 0; nx = 0
for line in open(rd + '/agentx/aiperf_artifacts/profile_export.jsonl'):
    try: j = json.loads(line)
    except ValueError: continue
    m = j['metadata']
    if m.get('benchmark_phase') != 'profiling': continue
    nprof += 1; ps.append(m['request_start_ns'] / 1e9)
    e = m['request_end_ns'] / 1e9
    if inw(e): nw += 1
    if x0 <= e < x1: nx += 1
print('profiling records', nprof, 'first start', datetime.fromtimestamp(min(ps), timezone.utc), 'n in window', nw, 'n in excl', nx)
rx = re.compile(r'^\[(\S+ \S+) PP0\] Prefill batch, (.*)')
ts = []
for line in open(rd + '/logs/yihou-llr-prefill0.log', errors='ignore'):
    mm = rx.match(line)
    if not mm: continue
    t = P(mm.group(1))
    d = dict((k.strip(), float(v)) for k, v in re.findall(r'(#[\w-]+): ([\d.]+)', mm.group(2)))
    ts.append((t, d))
span = (t1 - t0) - (x1 - x0)
w = [x for x in ts if inw(x[0])]
print('PP0 batches', len(w), 'per s %.3f' % (len(w) / span), 'new tok/s %.0f' % (sum(d['#new-token'] for _, d in w) / span))
for k in ['#queue-req', '#pending-token', '#bootstrap-req', '#inflight-req']:
    lw = sum(d[k] for _, d in w) / len(w)
    cf = []; i = 0; last = None; g = int(t0)
    while g < t1:
        while i < len(ts) and ts[i][0] <= g: last = ts[i][1][k]; i += 1
        if inw(g) and last is not None: cf.append(last)
        g += 1
    print('  %-15s line-weighted %.2f carry-forward %.2f' % (k, lw, sum(cf) / len(cf)))
rx = re.compile(r'^\[(\S+ \S+) DP(\d) TP\d\] Decode batch, #running-req: (\d+).*token usage: ([\d.]+).*#prealloc-req: (\d+), #transfer-req: (\d+)')
acc = {}
for line in open(rd + '/logs/yihou-llr-decode.log', errors='ignore'):
    mm = rx.match(line)
    if not mm or not inw(P(mm.group(1))): continue
    acc.setdefault(mm.group(2), []).append(tuple(map(float, mm.group(3, 4, 5, 6))))
for r in sorted(acc):
    a = acc[r]; n = len(a)
    hi = [x for x in a if x[1] >= .95]; lo = [x for x in a if x[1] < .95]
    print('DP%s n %d running %.2f usage %.3f prealloc>0 %.2f transfer %.2f | lw frac usage>=.95 %.3f running hi/lo %.1f/%.1f' % (
        r, n, sum(x[0] for x in a) / n, sum(x[1] for x in a) / n, sum(x[2] > 0 for x in a) / n, sum(x[3] for x in a) / n,
        len(hi) / n, sum(x[0] for x in hi) / max(len(hi), 1), sum(x[0] for x in lo) / max(len(lo), 1)))
