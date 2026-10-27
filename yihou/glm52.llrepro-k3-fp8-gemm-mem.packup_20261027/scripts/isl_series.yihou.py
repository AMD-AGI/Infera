#!/usr/bin/env python3
"""isl_series (task llrepro 2026-10-26): profiling-phase requests per 10-min bucket of request start: count, mean ISL,
p90 ISL, mean output tokens. usage: python3 -I isl_series.yihou.py <profile_export.jsonl> [bucket_min=10]"""
import json, sys
B = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
rows = []
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    md, m = d.get('metadata', {}), d.get('metrics', {})
    if md.get('benchmark_phase') != 'profiling' or 'input_sequence_length' not in m:
        continue
    rows.append((md['request_start_ns'], m['input_sequence_length']['value'], m.get('output_sequence_length', {}).get('value', 0)))
t0 = min(r[0] for r in rows)
bk = {}
for s, i, o in rows:
    bk.setdefault(int((s - t0) / 1e9 / 60 // B), []).append((i, o))
print('bucket(min) | n | ISL mean | ISL p90 | OSL mean | sum(ISL) k-tokens started')
for b in sorted(bk):
    v = bk[b]; isl = sorted(x[0] for x in v)
    print(f'{int(b * B):3d} | {len(v):5d} | {sum(isl) / len(v):8.0f} | {isl[int(.9 * len(isl))]:8.0f} | {sum(x[1] for x in v) / len(v):6.0f} | {sum(isl) / 1e3:9.0f}')
