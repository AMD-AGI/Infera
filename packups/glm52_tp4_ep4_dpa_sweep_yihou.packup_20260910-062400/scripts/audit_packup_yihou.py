#!/usr/bin/env python3
"""Read-only artifact verification. No original workspace or GPU required."""
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[1]
records = json.loads((root / 'provenance/source-files.json').read_text())['sources']
for entry in records:
    data = (root / entry['destination']).read_bytes()
    if entry['encoding'] == 'gzip':
        data = gzip.decompress(data)
    assert len(data) == entry['bytes']
    assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['destination']
full_count = 0
for path in sorted((root / 'evidence/iterations').iterdir()):
    if not (path / 'result_yihou.json').exists():
        continue
    result = json.loads((path / 'result_yihou.json').read_text())
    status = json.loads((path / 'launch_status.json').read_text())
    assert status['exit_code'] == 0
    ranks = [json.loads((path / f'rank_{i}_yihou.json').read_text()) for i in range(4)]
    reps = result['representative_ranks']
    assert sum(ranks[i]['useful_output_tokens'] for i in reps) == result['useful_output_tokens']
    assert all(all(v == result['verify_iterations'] for v in counts.values()) for counts in result['graph_execution_counts_by_rank'].values())
    assert math.isclose(result['output_tokens_per_second'], result['useful_output_tokens'] / result['elapsed_seconds'])
    assert math.isclose(result['effective_token_latency_ms_per_user'], result['elapsed_seconds'] * 1000 * result['batch_size'] / result['useful_output_tokens'])
    for file in (root / 'scripts/original/bench').glob('*.py'):
        assert file.read_bytes() == (path / 'bench_snapshot' / file.name).read_bytes()
    if path.name.startswith('full_'):
        full_count += 1
        assert result['complete'] and result['useful_output_tokens'] == result['batch_size'] * 10000
        assert result['final_seq_lens'] == [80000] * result['batch_size']
        assert result['tp_size'] == result['ep_size'] == 4
        assert result['dp_size'] == (4 if '_on_' in path.name else 1)
        assert result['input_len'] == 70000 and result['output_len'] == 10000
assert full_count == 10
for file in (root / 'scripts/original').rglob('*.py'):
    ast.parse(file.read_text())
for file in (root / 'scripts/original').rglob('*.sh'):
    subprocess.run(['bash', '-n', str(file)], check=True)
manifest = root / 'MANIFEST.sha256'
if manifest.exists():
    for line in manifest.read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest, name
print(json.dumps({'status': 'PASS', 'source_records': len(records), 'full_points': full_count, 'gpu_rerun': False}, indent=2))
