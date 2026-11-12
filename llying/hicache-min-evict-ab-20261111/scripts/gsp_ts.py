#!/usr/bin/env python3
"""Per-interval prefill rates in round 2 from prefill_ts.prom (pp_rank 0): computed / device-hit / host-hit tokens per s,
evictions and eviction seconds per interval. Usage: gsp_ts.py RUN_DIR [STEP=3]"""
import re, sys
from pathlib import Path
d = Path(sys.argv[1]); step = int(sys.argv[2]) if len(sys.argv) > 2 else 3
t0, t1 = float((d / 'r2_start').read_text()), float((d / 'r2_end').read_text())
snaps, cur = [], None
for line in (d / 'prefill_ts.prom').read_text().splitlines():
    if line.startswith('# '):
        cur = {'t': float(line[2:])}; snaps.append(cur); continue
    if 'pp_rank="0"' not in line or cur is None:
        continue
    name = line.split('{')[0]; mode = re.search(r'mode="(\w+)"', line)
    key = name + (':' + mode.group(1) if mode else '')
    cur[key] = cur.get(key, 0) + float(line.rsplit(' ', 1)[1])
snaps = [s for s in snaps if t0 - 10 <= s['t'] <= t1 + 10 and len(s) > 1]
K = ['sglang:prefill_effective_tokens_total:input', 'sglang:prefill_effective_tokens_total:device_hit',
     'sglang:prefill_effective_tokens_total:host_hit', 'sglang:eviction_duration_seconds_count',
     'sglang:eviction_duration_seconds_sum', 'sglang:num_queue_reqs', 'sglang:token_usage']
print('t(s)  comp/s  dev/s  host/s  evN  evSec  queue  usage')
for a, b in zip(snaps[::step], snaps[step::step]):
    dt = b['t'] - a['t']
    r = [(b.get(k, 0) - a.get(k, 0)) for k in K[:5]]
    print(f"{a['t'] - t0:5.0f} {r[0] / dt:7.0f} {r[1] / dt:7.0f} {r[2] / dt:6.0f} {r[3]:4.0f} {r[4]:5.2f} {b.get(K[5], 0):5.0f} {b.get(K[6], 0):.2f}")
