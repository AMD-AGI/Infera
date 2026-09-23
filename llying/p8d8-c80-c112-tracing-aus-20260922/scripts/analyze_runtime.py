#!/usr/bin/env python3
"""Summarize sampled resources by client phase and minute; retain rate semantics."""
import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from analyze import jsonl, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = args.run
    summary = json.loads((root / 'analysis/summary.json').read_text())
    intervals = []
    for c, point in summary['points'].items():
        for phase, data in point['phases'].items():
            start = data['start_ns']
            # The fixed profiling sending window excludes drain resource samples.
            end = start + 3600 * 10**9 if phase == 'profiling' else data['end_ns']
            if start and end:
                intervals.append((f'C{c}/{phase}', start, end))

    def cohort(ns):
        for name, start, end in intervals:
            if start <= ns < end:
                return name, int((ns - start) / 60e9)
        return None

    values = collections.defaultdict(list)
    windows = collections.defaultdict(list)
    counts = collections.Counter()

    def add(group, key, value):
        if group is not None:
            name, minute = group
            values[f'{name}/{key}'].append(value)
            windows[f'{name}/{minute}/{key}'].append(value)

    previous = {}
    for row in jsonl(root / 'sampling/engine.jsonl'):
        if row.get('record_type') != 'sample':
            continue
        ns = int(dt.datetime.fromisoformat(row['captured_at']).timestamp() * 1e9)
        group = cohort(ns)
        role = row['endpoint']
        if group: counts[f'{group[0]}/{role}/scrape_samples'] += 1
        if row.get('error'):
            counts[f'{role}/scrape_errors'] += 1
            if group: counts[f'{group[0]}/{role}/scrape_errors'] += 1
            continue
        gauge = collections.defaultdict(list)
        rates = collections.defaultdict(float)
        for series in row.get('series', []):
            metric, labels, value = series['metric'], series['labels'], series['value']
            if not isinstance(value, (int, float)):
                continue
            label_key = tuple(sorted(labels.items()))
            key = role, metric, label_key
            is_counter = metric.endswith('_total')
            if is_counter:
                old = previous.get(key)
                if old and ns > old[0] and value >= old[1]:
                    # Separate graph modes; avoid mixing eager and captured execution.
                    suffix = '/' + labels['mode'] if 'mode' in labels else ''
                    rates[metric + suffix] += (value - old[1]) / ((ns - old[0]) / 1e9)
                elif old and value < old[1]:
                    counts[f'{role}/{metric}/counter_resets'] += 1
                previous[key] = ns, value
            elif not metric.endswith(('_bucket', '_sum', '_count')):
                gauge[metric].append(value)
                if metric in ('sglang:num_queue_reqs', 'sglang:token_usage', 'sglang:num_running_reqs'):
                    add(group, f'{role}/rank{labels.get("dp_rank")}/{metric}', value)
        for metric, samples in gauge.items():
            add(group, f'{role}/{metric}/sum', sum(samples))
            add(group, f'{role}/{metric}/max_rank', max(samples))
        for metric, value in rates.items():
            add(group, f'{role}/{metric}/per_second', value)

    previous_nodes = {}
    for row in jsonl(root / 'sampling/nodes.jsonl'):
        if row.get('record_type') != 'sample':
            continue
        ns = int(dt.datetime.fromisoformat(row['captured_at']).timestamp() * 1e9)
        group, role = cohort(ns), row['role']
        if group: counts[f'{group[0]}/{role}/node_samples'] += 1
        if row.get('error'):
            counts[f'{role}/node_errors'] += 1
            continue
        sample = row.get('sample', {})
        for card, fields in sample.get('gpu', {}).get('json', {}).items():
            for metric in ('GPU use (%)', 'GPU Memory Read/Write Activity (%)'):
                try:
                    add(group, f'{role}/{card}/{metric}', float(fields[metric]))
                except (ValueError, KeyError):
                    pass
        for rail, fields in sample.get('hcas', {}).items():
            rail_counters = dict(fields.get('counters', {}), **fields.get('hw_counters', {}))
            for metric, value in rail_counters.items():
                if metric == 'lifespan':
                    continue
                if not isinstance(value, (int, float)):
                    continue
                key = role, rail, metric
                old = previous_nodes.get(key)
                if old and ns > old[0] and value >= old[1]:
                    delta = value - old[1]
                    if metric in ('port_xmit_data', 'port_rcv_data'):
                        add(group, f'{role}/{rail}/{metric}/GBps', delta * 4 / (ns - old[0]))
                    elif metric in ('tx_rdma_ucast_bytes', 'rx_rdma_ucast_bytes', 'tx_rdma_retx_bytes'):
                        add(group, f'{role}/{rail}/{metric}/GBps', delta / (ns - old[0]))
                    elif delta and group:
                        counts[f'{group[0]}/{role}/{rail}/{metric}/delta'] += delta
                previous_nodes[key] = ns, value

    # Host load acknowledgements share a scheduler monotonic clock with submits.
    host_events = []
    for path in (root / 'diagnostics/prefill').glob('*.jsonl'):
        host_events.extend(row for row in jsonl(path) if row.get('event') in
                           ('host_load_submitted', 'host_load_ack'))
    pending = {}
    for row in sorted(host_events, key=lambda x: (x['pid'], x['mono_ns'])):
        key = row['pid'], row['node_id']
        if row['event'] == 'host_load_submitted':
            if key in pending:
                counts['host_load/overwritten_submit'] += 1
            pending[key] = row
        else:
            start = pending.pop(key, None)
            if start is None:
                counts['host_load/unmatched_ack'] += 1
                continue
            elapsed = (row['mono_ns'] - start['mono_ns']) / 1e6
            add(cohort(start['wall_ns']), 'prefill/host_load_submit_to_cpu_ack_ms', elapsed)
    counts['host_load/unacknowledged_submits'] = len(pending)
    result = dict(intervals=intervals, accounting=dict(counts),
                  resources={k: stats(v) for k, v in values.items()},
                  minute_resources={k: stats(v) for k, v in windows.items()},
                  limitations=[
                      'Profiling resource windows are fixed 3600-second sending windows; drain excluded.',
                      'Counter rates use sample deltas; boundary samples can straddle a phase boundary.',
                      'Host ack is a CPU completion observation, not a pure DMA measurement.',
                      'Port data counters use the InfiniBand four-byte unit; GB/s is decimal.',
                      'Ionic hardware *_bytes counters use bytes directly; standard port data counters may be absent.',
                      'Metric sums are across exported rank series; use max_rank for fractional occupancy.'])
    (root / 'analysis/runtime-summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(root / 'analysis/runtime-summary.json')


if __name__ == '__main__':
    main()
