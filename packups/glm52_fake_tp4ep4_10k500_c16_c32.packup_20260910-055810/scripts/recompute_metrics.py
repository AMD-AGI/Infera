#!/usr/bin/env python3
"""NEW CPU-only recomputation from preserved full JSON and server logs.

Exact TPOT requires per-request E2E latency, absent in output-details JSON.
ITL sums provide an independently recomputed close estimate, not exact TPOT.
"""
import argparse
from collections import Counter
import gzip
import json
import math
from pathlib import Path
import re


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def compute(root):
    rows = []
    for name, conc in [('c16', 16), ('c32', 32)]:
        directory = root / 'rounds' / name
        data = json.loads((directory / 'benchmark.jsonl').read_text())
        assert data['completed'] == 128
        assert len(data['errors']) == 128 and not any(data['errors'])
        assert data['input_lens'] == [10000] * 128
        assert data['output_lens'] == [500] * 128
        assert len(data['ttfts']) == len(data['itls']) == len(data['generated_texts']) == 128
        assert all(len(itls) == 499 for itls in data['itls'])
        assert data['max_concurrency'] == conc
        outputs = sum(data['output_lens'])
        throughput = outputs / data['duration']
        assert outputs == data['total_output_tokens'] == 64000
        assert sum(data['input_lens']) == data['total_input_tokens'] == 1280000
        assert math.isclose(throughput, data['output_throughput'], rel_tol=1e-12)
        flat_itl = [v * 1000 for arr in data['itls'] for v in arr]
        itl_p50, itl_p90 = percentile(flat_itl, .5), percentile(flat_itl, .9)
        assert math.isclose(itl_p50, data['median_itl_ms'], abs_tol=1e-9)
        assert math.isclose(itl_p90, data['p90_itl_ms'], abs_tol=1e-9)
        reconstructed_tpot = [sum(arr) * 1000 / 499 for arr in data['itls']]
        tpot_estimate = percentile(reconstructed_tpot, .5)
        tpot_delta = tpot_estimate - data['median_tpot_ms']
        assert abs(tpot_delta) < .001
        reconstructed_concurrency = sum(t + sum(a) for t, a in zip(data['ttfts'], data['itls'])) / data['duration']
        assert abs(reconstructed_concurrency - data['concurrency']) < .001
        assert data['accept_length'] == data['server_info']['internal_states'][0]['avg_spec_accept_length']
        text = gzip.decompress((directory / 'server-full.log.gz').read_bytes()).decode()
        samples = [line for line in text.splitlines() if 'Decode batch,' in line]
        occupancy = Counter(int(re.search(r'#running-req: (\d+)', line)[1]) for line in samples)
        retractions = {int(re.search(r'#retracted-req: (\d+)', line)[1]) for line in samples}
        assert retractions == {0}
        assert all('cuda graph: True' in line for line in samples)
        if conc == 16:
            assert occupancy == {16: 114}
        else:
            assert occupancy == {32: 56, 16: 1}
            assert 'FlyDSL sparse MLA decode declined: seq 192, need 1..96' in text
        rows.append({
            'point': name, 'completed': data['completed'], 'errors': 0,
            'input_length': 10000, 'output_length': 500, 'duration_s': data['duration'],
            'output_tokens': outputs, 'output_tok_s_4gpu': throughput,
            'output_tok_s_per_gpu': throughput / 4,
            'p50_tpot_ms_reported': data['median_tpot_ms'],
            'p90_tpot_ms_reported': data['p90_tpot_ms'],
            'p50_tpot_ms_itl_estimate': tpot_estimate,
            'p50_tpot_estimate_minus_reported_ms': tpot_delta,
            'p50_itl_ms_recomputed': itl_p50, 'p90_itl_ms_recomputed': itl_p90,
            'accept_length_reported': data['accept_length'],
            'client_concurrency_reported': data['concurrency'],
            'client_concurrency_itl_estimate': reconstructed_concurrency,
            'sampled_running_counts': dict(sorted(occupancy.items())),
            'sampled_retracted_counts': sorted(retractions),
            'all_sampled_cuda_graph': True,
            'c32_192_row_flydsl_decline': conc == 32,
        })
    a, b = rows
    return {'points': rows, 'c32_vs_c16_percent': {
        'output_throughput': (b['output_tok_s_4gpu'] / a['output_tok_s_4gpu'] - 1) * 100,
        'p50_tpot': (b['p50_tpot_ms_reported'] / a['p50_tpot_ms_reported'] - 1) * 100,
        'p90_tpot': (b['p90_tpot_ms_reported'] / a['p90_tpot_ms_reported'] - 1) * 100,
    }, 'limits': [
        'TPOT recorded from (request latency - TTFT)/(output_len-1); per-request latency was not serialized. ITL-sum estimates differ by final response overhead.',
        'Acceptance cross-checked against embedded server_info, not independently reconstructed from raw acceptance counters (not saved).',
        'Scheduler counts are sampled observations including warmup, not time-weighted occupancy.',
        'One short run per point; synthetic input token throughput is not real served-prefill throughput.',
    ]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(compute(args.root), indent=2))
