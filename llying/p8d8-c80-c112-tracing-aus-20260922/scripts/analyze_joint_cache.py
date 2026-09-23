#!/usr/bin/env python3
"""Read-only cohort audit for joint cache/routing research; no token-level causality."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def source_key(row):
    meta = row['metadata']
    if meta.get('source_trace_id') is None or meta.get('source_outer_idx') is None:
        return None
    return tuple(meta.get(k) for k in (
        'source_trace_id', 'source_outer_idx', 'source_inner_idx', 'source_kind'))


def miss(row):
    p = row['prefill']
    return p['input_tokens'] - sum(p[k] for k in (
        'cached_device', 'cached_host', 'cached_storage'))


def summarize(rows):
    n = len(rows)
    return {
        'requests': n,
        'input_tokens': sum(r['prefill']['input_tokens'] for r in rows),
        'miss_tokens': sum(miss(r) for r in rows),
        'host_tokens': sum(r['prefill']['cached_host'] for r in rows),
        'p_queue_mean_s': sum(r['prefill']['durations_ms']['queue_ms'] for r in rows) / max(1, n) / 1000,
        'p_forward_mean_s': sum(r['prefill']['durations_ms']['forward_envelope_ms'] for r in rows) / max(1, n) / 1000,
        'd_wait_input_occupancy_proxy_tokens': sum(
            r['prefill']['input_tokens'] * r['decode']['durations_ms']['transfer_wait_ms']
            for r in rows) / 1000 / 3600,
        'd_generation_input_occupancy_proxy_tokens': sum(
            r['prefill']['input_tokens'] * r['decode']['durations_ms']['generation_ms']
            for r in rows) / 1000 / 3600,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    source = args.run / 'analysis/joined-requests.jsonl'
    data = source.read_bytes()
    all_rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    rows = [r for r in all_rows if r['phase'] == 'profiling']
    assert len({(r['concurrency'], r['rid']) for r in rows}) == len(rows)
    assert all(miss(r) >= 0 for r in rows)
    points = {}
    indexes = {}
    for c in (80, 112):
        cohort = [r for r in rows if r['concurrency'] == c]
        assert len(cohort) == {80: 9651, 112: 9321}[c]
        tail = [r for r in cohort if miss(r) >= 32768]
        groups = {'all': summarize(cohort), 'miss_ge32768': summarize(tail)}
        groups['tail_miss_share'] = groups['miss_ge32768']['miss_tokens'] / groups['all']['miss_tokens']
        groups['tail_d_wait_occupancy_share'] = groups['miss_ge32768']['d_wait_input_occupancy_proxy_tokens'] / groups['all']['d_wait_input_occupancy_proxy_tokens']
        groups['tail_by_p_rank'] = {
            str(rank): summarize([r for r in tail if r['prefill']['dp_rank'] == rank])
            for rank in range(8)}
        idx = collections.defaultdict(list)
        for r in cohort:
            key = source_key(r)
            if key is not None:
                idx[key].append(r)
        indexes[c] = idx
        groups['source_positions'] = {
            'distinct': len(idx),
            'duplicate_positions': sum(len(v) > 1 for v in idx.values()),
            'missing_position_requests': sum(source_key(r) is None for r in cohort),
        }
        points[str(c)] = groups
    common = set(indexes[80]) & set(indexes[112])
    unique_pairs = [(indexes[80][k][0], indexes[112][k][0]) for k in common
                    if len(indexes[80][k]) == len(indexes[112][k]) == 1]
    same_length = [(a, b) for a, b in unique_pairs
                   if a['prefill']['input_tokens'] == b['prefill']['input_tokens']]
    matched = {
        'common_source_positions': len(common),
        'unique_pairs': len(unique_pairs),
        'same_input_length_pairs': len(same_length),
        'same_input_length_rank_changed_pairs': sum(
            a['prefill']['dp_rank'] != b['prefill']['dp_rank'] for a, b in same_length),
        'same_length_cohorts': {
            '80': summarize([a for a, _ in same_length]),
            '112': summarize([b for _, b in same_length]),
        },
        'warning': 'Source position and equal input length do not prove identical token IDs, arrival order, or cache history. No causal routing attribution.',
    }
    output = {
        'source': str(source),
        'source_sha256': hashlib.sha256(data).hexdigest(),
        'method': 'Valid profiling cohort; tail threshold is actual miss>=32768; duration-weighted occupancy divided by 3600s, not window-clipped; output KV and page rounding excluded.',
        'points': points,
        'cross_concurrency_source_matching': matched,
    }
    args.output.write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({
        'points': {c: {k: v for k, v in p.items() if k != 'tail_by_p_rank'} for c, p in points.items()},
        'cross_concurrency_source_matching': matched,
    }, indent=2))


if __name__ == '__main__':
    main()
