#!/usr/bin/env python3
"""qseries (task llrepro 2026-10-24): decode/prefill queue counters per time bucket from a llying-kit run's logs.
usage: python3 -I qseries.yihou.py <run dir> <start HH:MM:SS> <end HH:MM:SS> [bucket_min=5]
Decode: per bucket, mean over log lines of the SUM over the 4 DP ranks (each rank's latest value carried within a 10 s
grid) for running / prealloc / transfer / queue / retracted, and mean / max token usage across ranks.
Prefill (PP0 lines): line-mean of queue / bootstrap / inflight / pending-token, plus new and cached tokens per second.
"""
import re, sys
from datetime import datetime, timedelta

run, t0s, t1s = sys.argv[1:4]
B = float(sys.argv[4]) if len(sys.argv) > 4 else 5.0
TS = re.compile(r'^\[(\d{4}-\d\d-\d\d) (\d\d:\d\d:\d\d) (\w+)')
KV = re.compile(r'([#\w\- ()/]+?): ([\d.]+)(?:,|$)')


def rows(path, key):
    with open(path, errors='replace') as f:
        for line in f:
            if key in line:
                m = TS.match(line)
                if m:
                    t = datetime.strptime(m.group(1) + ' ' + m.group(2), '%Y-%m-%d %H:%M:%S')
                    yield t, m.group(3), {k.strip(): float(v) for k, v in KV.findall(line.split(key, 1)[1])}


dec = list(rows(f'{run}/logs/yihou-llr-decode.log', 'Decode batch,'))
pre = [r for r in rows(f'{run}/logs/yihou-llr-prefill0.log', 'Prefill batch,') if r[1] == 'PP0']
day = dec[len(dec) // 2][0].date()
t0 = datetime.combine(day, datetime.strptime(t0s, '%H:%M:%S').time())
t1 = datetime.combine(day, datetime.strptime(t1s, '%H:%M:%S').time())
if t1 < t0:
    t1 += timedelta(days=1)
nb = int(((t1 - t0).total_seconds() / 60) // B)

# decode: 10 s grid, carry each rank's last value forward, sum over ranks
grid = {}
last = {}
for t, r, d in sorted(dec, key=lambda x: x[0]):
    last[r] = d
    if t0 <= t < t1 and len(last) == 4:
        grid[int((t - t0).total_seconds() // 10)] = {k: [last[x] for x in sorted(last)] for k in ('_',)}['_']
print(f'{run.rstrip("/").split("/")[-1]}  window {t0:%H:%M:%S}-{t1:%H:%M:%S}  bucket {B:g} min')
print('DECODE (sum of 4 ranks): bucket | running | prealloc | transfer | queue | retracted | usage mean/max-rank'
      ' | rank running min-max')
for b in range(nb):
    g = [v for k, v in grid.items() if b * B * 6 <= k < (b + 1) * B * 6]
    if not g:
        continue
    s = lambda key: sum(sum(x.get(key, 0) for x in v) for v in g) / len(g)
    um = sum(sum(x.get('token usage', 0) for x in v) / 4 for v in g) / len(g)
    ux = sum(max(x.get('token usage', 0) for x in v) for v in g) / len(g)
    rr = [sum(v[i].get('#running-req', 0) for v in g) / len(g) for i in range(4)]
    st = t0 + timedelta(minutes=b * B)
    print(f'{st:%H:%M} | {s("#running-req"):5.1f} | {s("#prealloc-req"):4.1f} | {s("#transfer-req"):5.1f} |'
          f' {s("#queue-req"):4.1f} | {s("#retracted-req"):3.1f} | {um:.2f}/{ux:.2f} | {min(rr):4.1f}-{max(rr):4.1f}')
print('PREFILL PP0 (line mean): bucket | queue | bootstrap | inflight | pending-tok(k) | new tok/s | cached tok/s'
      ' | batches/s')
for b in range(nb):
    a, z = t0 + timedelta(minutes=b * B), t0 + timedelta(minutes=(b + 1) * B)
    g = [d for t, _, d in pre if a <= t < z]
    if not g:
        continue
    m = lambda key: sum(d.get(key, 0) for d in g) / len(g)
    sec = B * 60
    print(f'{a:%H:%M} | {m("#queue-req"):4.1f} | {m("#bootstrap-req"):4.1f} | {m("#inflight-req"):4.1f} |'
          f' {m("#pending-token") / 1e3:6.0f} | {sum(d.get("#new-token", 0) for d in g) / sec:7.0f} |'
          f' {sum(d.get("#cached-token", 0) for d in g) / sec:7.0f} | {len(g) / sec:.2f}')
