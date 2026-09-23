#!/usr/bin/env python3
"""Measure simultaneous rank skew from complete scrapes; not routing benefit."""
import argparse
import collections
import datetime as dt
import json
from pathlib import Path

from analyze import jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.run / 'analysis/summary.json').read_text())
    intervals = [(c, p['phases']['profiling']['start_ns'])
                 for c, p in summary['points'].items()]
    counts = collections.defaultdict(collections.Counter)
    for row in jsonl(args.run / 'sampling/engine.jsonl'):
        if row.get('error') or 'captured_at' not in row:
            continue
        ns = int(dt.datetime.fromisoformat(row['captured_at']).timestamp() * 1e9)
        c = next((c for c, start in intervals if start <= ns < start + 3600e9), None)
        role = row.get('endpoint')
        if c is None or role not in ('prefill', 'decode'):
            continue
        metrics = collections.defaultdict(dict)
        duplicates = set()
        for series in row.get('series', []):
            rank = series.get('labels', {}).get('dp_rank')
            metric = series['metric']
            if rank is not None:
                if rank in metrics[metric]:
                    duplicates.add(metric)
                metrics[metric][rank] = series['value']
        names = ['sglang:num_queue_reqs', 'sglang:num_running_reqs'] if role == 'prefill' else ['sglang:token_usage']
        out = counts[f'C{c}/{role}']
        out['successful_scrapes'] += 1
        if any(n in duplicates or set(metrics[n]) != set(map(str, range(8))) for n in names):
            out['incomplete_or_duplicate_rank_scrapes'] += 1
            continue
        out['complete_rank_scrapes'] += 1
        if role == 'prefill':
            queue, running = (metrics[n] for n in names)
            if max(queue.values()) > 0:
                out['any_queue'] += 1
                if min(queue.values()) == 0:
                    out['queue_and_other_rank_zero_queue'] += 1
                if any(queue[r] == 0 and running[r] == 0 for r in queue):
                    out['queue_and_other_rank_zero_queue_zero_running'] += 1
            if min(queue.values()) > 0:
                out['all_ranks_queued'] += 1
        else:
            usage = list(metrics[names[0]].values())
            if max(usage) >= .9:
                out['some_rank_kv_ge_90pct'] += 1
                if min(usage) < .7:
                    out['kv_ge_90pct_and_other_rank_lt_70pct'] += 1
    result = {'counts': dict(counts), 'limitations': [
        'Sample counts, not time-weighted durations; errors and incomplete rank scrapes excluded.',
        'Zero queue/running does not establish GPU idle or available TP collective capacity.',
        'KV skew does not establish that another rank can admit a specific blocked request.',
        'Thresholds are descriptive (90%/70%), not validated admission constraints.',
        'Profiling window starts at earliest exported request and lasts 3600 seconds.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
