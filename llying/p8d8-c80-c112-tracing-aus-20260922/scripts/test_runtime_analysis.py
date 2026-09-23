#!/usr/bin/env python3
"""Check rate units, phase bounds and monotonic host load acknowledgement."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    for name in ('analysis', 'sampling', 'diagnostics/prefill'):
        (root / name).mkdir(parents=True)
    (root / 'analysis/summary.json').write_text(json.dumps({'points': {'80': {'phases': {
        'profiling': {'start_ns': 1_000_000_000, 'end_ns': 4_000_000_000_000}}}}}))
    engines, nodes = [], []
    for sec, value in [(1, 10), (3, 30)]:
        base = dict(record_type='sample', captured_at=f'1970-01-01T00:00:0{sec}+00:00')
        engines.append(dict(base, endpoint='decode', series=[dict(metric='sglang:generation_tokens_total',
            labels={'dp_rank': '0'}, value=value)]))
        nodes.append(dict(base, role='prefill', sample={'hcas': {'ionic_0': {
            'counters': {'port_xmit_data': value * 250_000_000},
            'hw_counters': {'tx_rdma_ucast_bytes': value * 1_000_000_000}}}}))
    for name, rows in [('engine', engines), ('nodes', nodes)]:
        (root / f'sampling/{name}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    host = [dict(event=event, pid=1, node_id=2, wall_ns=wall, mono_ns=mono)
            for event, wall, mono in [('host_load_submitted', 2_000_000_000, 100),
                                      ('host_load_ack', 1_500_000_000, 10_000_100)]]
    (root / 'diagnostics/prefill/1.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in host))
    subprocess.run([sys.executable, str(Path(__file__).with_name('analyze_runtime.py')), str(root)], check=True)
    data = json.loads((root/'analysis/runtime-summary.json').read_text())
    resources = data['resources']
    assert resources['C80/profiling/decode/sglang:generation_tokens_total/per_second']['p50'] == 10
    assert resources['C80/profiling/prefill/ionic_0/port_xmit_data/GBps']['p50'] == 10
    assert resources['C80/profiling/prefill/ionic_0/tx_rdma_ucast_bytes/GBps']['p50'] == 10
    assert resources['C80/profiling/prefill/host_load_submit_to_cpu_ack_ms']['p50'] == 10
    assert data['intervals'][0][2] == 3_601_000_000_000
    print('PASS: counter rates, four-byte units, phase duration and monotonic host ack')
