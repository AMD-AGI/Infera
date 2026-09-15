#!/usr/bin/env python3
"""Validate native results and pair them with the sealed internal baseline."""
import csv
import io
import json
from pathlib import Path

W = Path(__file__).resolve().parents[1]


def validate(record, mode, concurrency):
    expected = {'completed': 128, 'total_input_tokens': 128 * 70000,
        'total_output_tokens': 128 * 10000, 'random_input_len': 70000,
        'random_output_len': 10000, 'max_concurrency': concurrency}
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'{key}: expected {value}, got {record.get(key)}')
    if record.get('input_lens') != [70000] * 128:
        raise ValueError('Per-request input lengths differ from70000')
    if record.get('output_lens') != [10000] * 128:
        raise ValueError('Per-request output lengths differ from10000')
    info = record['server_info']
    for key, value in {'tp_size': 4, 'ep_size': 4, 'dp_size': 4 if mode == 'on' else 1,
            'enable_dp_attention': mode == 'on', 'max_running_requests': concurrency,
            'disaggregation_mode': 'decode', 'disaggregation_transfer_backend': 'fake',
            'speculative_algorithm': 'EAGLE', 'speculative_num_steps': 5,
            'speculative_eagle_topk': 1, 'speculative_num_draft_tokens': 6,
            'kv_cache_dtype': 'fp8_e4m3', 'disable_overlap_schedule': False}.items():
        if info.get(key) != value:
            raise ValueError(f'server_info.{key}: expected {value}, got {info.get(key)}')
    reconstructed = record['total_output_tokens'] / record['duration']
    if abs(reconstructed / record['output_throughput'] - 1) > 1e-8:
        raise ValueError('Throughput does not match raw tokens/duration')
    return {'request_lengths_verified': True, 'topology_verified': True,
            'accept_length': record.get('accept_length'),
            'effective_concurrency': record.get('concurrency')}


def main():
    baseline = list(csv.DictReader((W / 'reference/internal_summary.csv').open()))
    rows = []
    lines = ['# Reverse fake-server sweep comparison', '',
        'Native server:128 rolling requests,70K/10K,16 short warmups. Internal:one fixed batch.',
        'TPOT delta=(server mean/internal mean-1). Throughput delta=(server/internal-1).', '',
        '| DPA | C | Status | Internal TPOT ms | Server mean / P50 / P90 ms | TPOT delta | Throughput internal / server | Throughput delta |',
        '|---|---:|---|---:|---|---:|---|---:|']
    for ref in baseline:
        mode, c = ref['dpa'], int(ref['concurrency'])
        directory = W / 'rounds' / f'dpa_{mode}_c{c}_yihou'
        status = json.loads((directory / 'status.json').read_text()) if (directory / 'status.json').exists() else {'state': 'not_run'}
        row = {'dpa': mode, 'concurrency': c, 'status': status['state'],
            'internal_tpot_ms': float(ref['tpot_ms']),
            'internal_tps': float(ref['output_tokens_per_second'])}
        if status['state'] == 'passed':
            d = json.loads((directory / 'benchmark.jsonl').read_text().splitlines()[-1])
            verification = validate(d, mode, c)
            (directory / 'verification_yihou.json').write_text(json.dumps(verification, indent=2) + '\n')
            row.update(server_mean_tpot_ms=d['mean_tpot_ms'], server_p50_tpot_ms=d['median_tpot_ms'],
                server_p90_tpot_ms=d['p90_tpot_ms'], server_tps=d['output_throughput'],
                server_duration=d['duration'], accept_length=d.get('accept_length'),
                effective_concurrency=d.get('concurrency'), launch_wall_seconds=status['wall_seconds'])
            row['tpot_delta_pct'] = 100 * (row['server_mean_tpot_ms'] / row['internal_tpot_ms'] - 1)
            row['tps_delta_pct'] = 100 * (row['server_tps'] / row['internal_tps'] - 1)
            lines.append(f"| {mode} | {c} | pass | {row['internal_tpot_ms']:.3f} | {d['mean_tpot_ms']:.3f} / {d['median_tpot_ms']:.3f} / {d['p90_tpot_ms']:.3f} | {row['tpot_delta_pct']:+.1f}% | {row['internal_tps']:.1f} / {row['server_tps']:.1f} | {row['tps_delta_pct']:+.1f}% |")
        else:
            lines.append(f"| {mode} | {c} | {status['state']} | {row['internal_tpot_ms']:.3f} | — | — | {row['internal_tps']:.1f} / — | — |")
        rows.append(row)
    output = W / 'results'
    output.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader(); writer.writerows(rows)
    (output / 'summary.csv').write_text(buffer.getvalue())
    lines += ['', '## Scope and caveats',
        '- Close timings do not validate tokens,hidden states,real P-to-D transfer,or production acceptance.',
        '- Server TPOT=(latency-TTFT)/(OSL-1); internal TPOT=complete decode loop/OSL.',
        '- Server overlap,rolling refill/tail and recycled fake prefix differ from synchronous internal random prefix.',
        '-128 requests retain non-full tails atC20/C24; effective concurrency is recorded in CSV.',
        '- Hardware node271 differs from internal node056. Cold startup is excluded from native measurements.',
        '- FlyDSL configured backend does not guarantee every phase uses it. DPAon64Qheads and offC20/C24verify rows exceed the pinned FlyDSL gate; fallback remains unchanged.',
        '- Earlier eager/graph differences remain unresolved; user stopped that investigation.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    print(buffer.getvalue())


if __name__ == '__main__':
    main()
