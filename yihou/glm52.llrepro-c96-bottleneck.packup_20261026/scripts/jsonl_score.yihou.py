#!/usr/bin/env python3
"""jsonl_score (task llrepro 2026-10-25): recompute the AgentX score from aiperf profile_export.jsonl when
agentx_conc*.json is missing (NFS quota). score = (sum input + output tokens of profiling-phase records)
/ (last request_end - first request_start) / n_gpus. Also TTFT p50/p90 and mean ISL. Validate on a run that has the json.
usage: python3 -I jsonl_score.yihou.py <profile_export.jsonl> [n_gpus=8]"""
import json, sys

n_gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 8
tok = 0; t0 = None; t1 = None; ttft = []; isl = []; n = 0; bad = 0
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        bad += 1; continue
    m, md = d.get('metrics', {}), d.get('metadata', {})
    if md.get('benchmark_phase') != 'profiling' or md.get('was_cancelled'):
        continue
    i = m.get('input_sequence_length', {}).get('value'); o = m.get('output_sequence_length', {}).get('value')
    if i is None or o is None:
        continue
    n += 1; tok += i + o; isl.append(i)
    s, e = md['request_start_ns'], md['request_end_ns']
    t0 = s if t0 is None else min(t0, s); t1 = e if t1 is None else max(t1, e)
    if 'time_to_first_token' in m:
        ttft.append(m['time_to_first_token']['value'] / 1000)
ttft.sort()
q = lambda p: ttft[min(len(ttft) - 1, int(p * len(ttft)))]
span = (t1 - t0) / 1e9
print(f'records {n} (bad lines {bad}) span {span:.0f} s  score {tok / span / n_gpu:.0f}  isl mean {sum(isl) / len(isl):.0f}'
      f'  ttft p50/p90 {q(.5):.2f}/{q(.9):.2f}')
