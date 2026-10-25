#!/usr/bin/env python3
"""ts (llr7r analysis 2026-10-24): request accounting + decode KV-full vs admission + prefill time series.

usage: python3 -I ts.yihou.py <run dir> <start 'YYYY-mm-dd HH:MM:SS'> <end> [ex_start ex_end]
Read-only. Log timestamps are 1 s resolution; server gauges are carried forward on a 1 s grid between
log lines (prefill PP0 logs only when it runs a batch, so its gauges are stale during idle seconds).
"""
import re, sys, json, glob, statistics as st
from datetime import datetime, timezone

run = sys.argv[1]
P = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
t0, t1 = P(sys.argv[2]), P(sys.argv[3])
ex = (P(sys.argv[4]), P(sys.argv[5])) if len(sys.argv) > 5 else None
grid = [t for t in range(int(t0), int(t1) + 1) if not (ex and ex[0] <= t <= ex[1])]
TS = re.compile(r'^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (\w+)(?: (TP\d+))?\]')
KV = re.compile(r'([#\w\- ()/]+?): ([\d.]+|True|False)(?:,|$)')


def parse(path, key):
    out = []
    with open(path, errors='replace') as fh:
        for line in fh:
            if key not in line:
                continue
            m = TS.match(line)
            if not m:
                continue
            t = P(m.group(1))
            if t < t0 - 60 or t > t1 + 60:
                continue
            d = {}
            for k, v in KV.findall(line.split(key, 1)[1]):
                try:
                    d[k.strip()] = float(v)
                except ValueError:
                    pass
            out.append((t, m.group(2), d))
    return out


def carry(rows, keys):
    """1 s grid, last value at or before t."""
    rows = sorted(rows, key=lambda x: x[0])
    res = {k: [] for k in keys}
    i, last = 0, None
    for t in grid:
        while i < len(rows) and rows[i][0] <= t:
            last = rows[i][1]; i += 1
        for k in keys:
            res[k].append(last.get(k, float('nan')) if last else float('nan'))
    return res


def m(xs):
    xs = [x for x in xs if x == x]
    return st.mean(xs) if xs else float('nan')


def pear(a, b):
    pr = [(x, y) for x, y in zip(a, b) if x == x and y == y]
    if len(pr) < 3:
        return float('nan')
    ma, mb = st.mean(p[0] for p in pr), st.mean(p[1] for p in pr)
    sa = sum((p[0] - ma) ** 2 for p in pr) ** .5; sb = sum((p[1] - mb) ** 2 for p in pr) ** .5
    return sum((p[0] - ma) * (p[1] - mb) for p in pr) / (sa * sb) if sa and sb else float('nan')


print(f'window {sys.argv[2]} .. {sys.argv[3]} excl {sys.argv[4:6]} grid {len(grid)} s')
dec = parse(f'{run}/logs/yihou-llr-decode.log', 'Decode batch,')
pre = parse(f'{run}/logs/yihou-llr-prefill0.log', 'Prefill batch,')
ranks = sorted({r for _, r, _ in dec})
DK = ['#running-req', '#prealloc-req', '#transfer-req', '#queue-req', 'token usage', 'pre-allocated usage', '#retracted-req']
dser = {r: carry([(t, d) for t, rr, d in dec if rr == r], DK) for r in ranks}
pp0 = [(t, d) for t, rr, d in pre if rr == 'PP0']
PK = ['#queue-req', '#pending-token', '#bootstrap-req', '#inflight-req', '#running-req', '#optimistic-req']
pser = carry(pp0, PK)

# ---- 3) request accounting ----
print('\n== 3) request accounting (time-averaged over the 1 s grid) ==')
recs = []
fp = glob.glob(f'{run}/agentx/**/profile_export.jsonl', recursive=True)
if fp:
    with open(fp[0]) as fh:
        for line in fh:
            try:
                j = json.loads(line)
            except ValueError:
                continue
            md, mt = j.get('metadata', {}), j.get('metrics', {})
            s, e = md.get('request_start_ns'), md.get('request_end_ns')
            if s is None or e is None:
                continue
            tt = mt.get('time_to_first_token', {}).get('value')
            recs.append((s / 1e9, e / 1e9, s / 1e9 + tt / 1e3 if tt is not None else e / 1e9,
                         md.get('credit_issued_ns', s) / 1e9))
print(f'client records {len(recs)} (file {fp[0] if fp else None})')
if recs:
    import bisect
    ev = []
    for s, e, ft, ci in recs:
        ev += [(s, 'inflight', 1), (e, 'inflight', -1), (s, 'prefirst', 1), (ft, 'prefirst', -1)]
    ev.sort()
    cur = {'inflight': 0, 'prefirst': 0}; i = 0; inf = []; pf = []
    for t in grid:
        while i < len(ev) and ev[i][0] <= t:
            cur[ev[i][1]] += ev[i][2]; i += 1
        inf.append(cur['inflight']); pf.append(cur['prefirst'])
    print(f'client in-flight (start..end) mean {m(inf):.1f} p10 {sorted(inf)[len(inf)//10]} p50 {sorted(inf)[len(inf)//2]}'
          f' max {max(inf)} ; of which pre-first-token mean {m(pf):.1f}, streaming (post-first-token) {m(inf) - m(pf):.1f}')
    print(f'client not in flight (80 - in-flight; AgentX idle/think/tool replay, credit gaps) mean {80 - m(inf):.1f}')
    gaps = [s - ci for s, e, ft, ci in recs if t0 <= s <= t1]
    if gaps:
        print(f'credit_issued -> request_start gap s: mean {m(gaps):.3f} max {max(gaps):.3f}')
sumr = lambda k: [sum(dser[r][k][i] for r in ranks) for i in range(len(grid))]
print('decode (sum over ranks):', ', '.join(f'{k} {m(sumr(k)):.1f}' for k in ['#running-req', '#prealloc-req', '#transfer-req', '#queue-req', '#retracted-req']))
print('prefill PP0 (carry-forward):', ', '.join(f'{k} {m(pser[k]):.1f}' for k in ['#queue-req', '#bootstrap-req', '#running-req', '#inflight-req', '#optimistic-req']))
print('note: prefill inflight-req (sent, awaiting transfer completion) and decode transfer-req/prealloc-req can refer to the SAME request; not additive.')

# ---- 4) decode KV-full vs admission ----
print('\n== 4) per decode rank: token usage >= thr vs prealloc / transfer (1 s grid, carry-forward) ==')
for thr in (0.95, 0.90):
    print(f'-- thr {thr}')
    print('rank | frac(tu>=thr) | P(prealloc>0|hi) P(prealloc>0|lo) | mean prealloc hi/lo | mean transfer hi/lo |'
          ' mean running hi/lo | frac of prealloc>0 time with hi | corr(tu,prealloc) corr(tu,transfer) corr(tu+pre-alloc usage,prealloc)')
    for r in ranks:
        d = dser[r]; tu = d['token usage']; pa = d['#prealloc-req']; tr = d['#transfer-req']; rn = d['#running-req']
        pu = d['pre-allocated usage']
        hi = [i for i in range(len(grid)) if tu[i] == tu[i] and tu[i] >= thr]
        lo = [i for i in range(len(grid)) if tu[i] == tu[i] and tu[i] < thr]
        pr = lambda idx, s: (sum(s[i] > 0 for i in idx) / len(idx)) if idx else float('nan')
        mm = lambda idx, s: m([s[i] for i in idx])
        pa_pos = [i for i in range(len(grid)) if pa[i] > 0]
        print(f'{r:4} | {len(hi) / len(grid):.3f} | {pr(hi, pa):.2f} {pr(lo, pa):.2f} | {mm(hi, pa):.2f} {mm(lo, pa):.2f} |'
              f' {mm(hi, tr):.2f} {mm(lo, tr):.2f} | {mm(hi, rn):.1f} {mm(lo, rn):.1f} |'
              f' {(sum(tu[i] >= thr for i in pa_pos) / len(pa_pos)) if pa_pos else float("nan"):.2f} |'
              f' {pear(tu, pa):.2f} {pear(tu, tr):.2f} {pear([a + b for a, b in zip(tu, pu)], pa):.2f}')
print('token usage quantiles per rank (p50/p90/max) and pre-allocated usage mean:')
for r in ranks:
    tu = sorted(x for x in dser[r]['token usage'] if x == x)
    print(f'  {r}: {tu[len(tu)//2]:.2f}/{tu[int(.9*len(tu))]:.2f}/{tu[-1]:.2f} ; pre-alloc usage mean {m(dser[r]["pre-allocated usage"]):.3f}'
          f' ; tu+prealloc usage >=0.95 frac {sum((a + b) >= .95 for a, b in zip(dser[r]["token usage"], dser[r]["pre-allocated usage"])) / len(grid):.3f}')

# ---- 5) prefill time series ----
print('\n== 5) prefill (PP0) ==')
sec = {}
for t, d in pp0:
    if t0 <= t <= t1 and not (ex and ex[0] <= t <= ex[1]):
        a = sec.setdefault(int(t), [0, 0, 0]); a[0] += 1; a[1] += d.get('#new-token', 0); a[2] += d.get('#cached-token', 0)
nb = sum(a[0] for a in sec.values())
print(f'PP0 batches {nb} over {len(grid)} s: {nb / len(grid):.2f}/s, mean interval {len(grid) / max(nb, 1) * 1000:.0f} ms;'
      f' seconds with >=1 PP0 batch {len(sec) / len(grid):.3f}; batches/s in active seconds p50 {sorted(a[0] for a in sec.values())[len(sec)//2] if sec else 0}')
newt = sum(a[1] for a in sec.values()); cact = sum(a[2] for a in sec.values())
print(f'new tok/s {newt / len(grid):.0f}, cached tok/s {cact / len(grid):.0f}; new tokens per batch mean {newt / max(nb, 1):.0f}')
qv = pser['#queue-req']
print(f'frac of time queue-req>0 (carry-forward) {sum(x > 0 for x in qv if x == x) / len(grid):.3f};'
      f' line-weighted {sum(d.get("#queue-req", 0) > 0 for t, d in pp0 if t0 <= t <= t1) / max(1, sum(1 for t, d in pp0 if t0 <= t <= t1)):.3f}')
print('per-minute buckets (stats over minutes): new tok/s | cached tok/s | batches/min | queue-req mean | pending-token mean | inflight mean | bootstrap mean')
bk = {}
for i, t in enumerate(grid):
    b = bk.setdefault(int((t - t0) // 60), {'n': 0, 'new': 0, 'ca': 0, 'nb': 0, 'q': [], 'pt': [], 'inf': [], 'bs': []})
    b['n'] += 1
    a = sec.get(t)
    if a:
        b['nb'] += a[0]; b['new'] += a[1]; b['ca'] += a[2]
    b['q'].append(pser['#queue-req'][i]); b['pt'].append(pser['#pending-token'][i])
    b['inf'].append(pser['#inflight-req'][i]); b['bs'].append(pser['#bootstrap-req'][i])
rows = [(b['new'] / b['n'], b['ca'] / b['n'], b['nb'] * 60 / b['n'], m(b['q']), m(b['pt']), m(b['inf']), m(b['bs']))
        for k, b in sorted(bk.items()) if b['n'] >= 30]
for j, name in enumerate(['new tok/s', 'cached tok/s', 'batches/min', 'queue-req', 'pending-token', 'inflight', 'bootstrap']):
    v = sorted(r[j] for r in rows if r[j] == r[j])
    if v:
        print(f'  {name:14} min {v[0]:10.1f} p10 {v[len(v)//10]:10.1f} p50 {v[len(v)//2]:10.1f} p90 {v[int(.9*len(v))]:10.1f} max {v[-1]:10.1f} cv {st.pstdev(v) / st.mean(v) if st.mean(v) else 0:.2f}')
print('5-min buckets: start(min) new tok/s, queue-req, pending-token(k), inflight, decode running sum, decode prealloc sum, decode transfer sum, decode tokusage mean')
for k5 in range(0, len(rows) and max(bk) + 1, 5):
    idx = [i for i, t in enumerate(grid) if k5 <= (t - t0) // 60 < k5 + 5]
    if len(idx) < 60:
        continue
    act = [sec.get(grid[i], [0, 0, 0])[1] for i in idx]
    print(f'  {k5:4d} {sum(act) / len(idx):8.0f} {m([pser["#queue-req"][i] for i in idx]):5.1f} {m([pser["#pending-token"][i] for i in idx]) / 1e3:7.0f}'
          f' {m([pser["#inflight-req"][i] for i in idx]):5.1f} {m([sum(dser[r]["#running-req"][i] for r in ranks) for i in idx]):5.1f}'
          f' {m([sum(dser[r]["#prealloc-req"][i] for r in ranks) for i in idx]):5.1f} {m([sum(dser[r]["#transfer-req"][i] for r in ranks) for i in idx]):5.1f}'
          f' {m([m([dser[r]["token usage"][i] for r in ranks]) for i in idx]):.3f}')
