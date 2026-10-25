#!/usr/bin/env python3
"""win (llr7r analysis 2026-10-24): client-side completions in a window from aiperf profile_export.jsonl.
usage: python3 -I win.yihou.py <jsonl> <start> <end> [ex_start ex_end]   (times 'YYYY-mm-dd HH:MM:SS' UTC)
Counts records whose request_end falls in the window (minus exclusion): req/s, ISL/OSL means, in/out tok/s, TTFT p50/p90, TPOT mean."""
import sys, json
from datetime import datetime, timezone
P = lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
t0, t1 = P(sys.argv[2]), P(sys.argv[3]); ex = (P(sys.argv[4]), P(sys.argv[5])) if len(sys.argv) > 5 else None
span = t1 - t0 - ((ex[1] - ex[0]) if ex else 0)
isl, osl, ttft, itl, keys = [], [], [], [], set()
for line in open(sys.argv[1]):
    try: j = json.loads(line)
    except ValueError: continue
    md, mt = j['metadata'], j['metrics']; keys |= set(mt)
    e = md.get('request_end_ns', 0) / 1e9
    if not (t0 <= e <= t1) or (ex and ex[0] <= e <= ex[1]) or md.get('benchmark_phase') != 'profiling': continue
    g = lambda k: (mt.get(k) or {}).get('value')
    isl.append(g('input_sequence_length') or 0); osl.append(g('output_sequence_length') or 0)
    if g('time_to_first_token') is not None: ttft.append(g('time_to_first_token') / 1e3)
    if g('inter_token_latency') is not None: itl.append(g('inter_token_latency'))
q = lambda xs, p: sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float('nan')
n = len(isl)
print(f'span {span:.0f} s n {n} req/s {n / span:.3f} ISL mean {sum(isl) / max(n,1):.0f} OSL mean {sum(osl) / max(n,1):.0f}'
      f' in tok/s/GPU(8) {sum(isl) / span / 8:.0f} out tok/s/GPU(8) {sum(osl) / span / 8:.0f}'
      f' TTFT p50/p90 {q(ttft, .5):.2f}/{q(ttft, .9):.2f} s ITL mean {sum(itl) / max(len(itl),1):.1f} ms')
if not isl or 'input_sequence_length' not in keys: print('metric keys:', sorted(keys))
