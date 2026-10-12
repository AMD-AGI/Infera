#!/usr/bin/env python3
"""First warmup wave per AgentX run: the first N warmup requests by credit_issued_ns.
Usage: firstwave.yihou.py N LABEL:profile_export.jsonl [...]"""
import json, sys, statistics as st
n = int(sys.argv[1]); waves = {}
def q(xs, p): xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]
print(f"{'run':6} {'n':>3} {'wave_s':>7} {'ttft_mean':>9} {'ttft_p50':>8} {'ttft_p90':>8} {'ttft_max':>8} {'isl_sum':>9} {'isl_mean':>8} {'agg_tok/s':>9}")
for spec in sys.argv[2:]:
    label, path = spec.split(':')
    recs = [json.loads(l) for l in open(path)]
    w = sorted((r for r in recs if r['metadata'].get('benchmark_phase') == 'warmup'), key=lambda r: r['metadata']['credit_issued_ns'])[:n]
    t0 = min(r['metadata']['request_start_ns'] for r in w); t1 = max(r['metadata']['request_end_ns'] for r in w)
    tt = [r['metrics']['time_to_first_token']['value'] / 1e3 for r in w if 'time_to_first_token' in r['metrics']]
    isl = [r['metrics']['input_sequence_length']['value'] for r in w if 'input_sequence_length' in r['metrics']]
    bad = sum('time_to_first_token' not in r['metrics'] for r in w)
    waves[label] = {r['metadata']['conversation_id'] for r in w}
    print(f"{label:6} {len(w):3d} {(t1-t0)/1e9:7.0f} {st.mean(tt):9.1f} {q(tt,.5):8.1f} {q(tt,.9):8.1f} {max(tt):8.1f} {sum(isl):9d} {st.mean(isl):8.0f} {sum(isl)/((t1-t0)/1e9):9.0f}  no_ttft={bad}")
labels = list(waves); base = labels[0]
for other in labels[1:]:
    common = set(waves[base]) & set(waves[other])
    print(f"same conversations {base} vs {other}: {len(common)}/{n}")
