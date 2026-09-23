#!/usr/bin/env python3
"""Check the generated benchmark settings and cached workload before load."""
import hashlib
import json
import os
from pathlib import Path
import sys

def env(path):
    return dict(line.split('=', 1) for line in path.read_text().splitlines()
                if line and not line.startswith('#') and '=' in line)

out = Path(sys.argv[1])
base = Path(os.environ['BASELINE_RUN'])
a, b = env(base/'c80/runtime.env'), env(out/'runtime.env')
path_keys = {'AGENTIC_OUTPUT_DIR', 'AIPERF_RUNTIME_DIR', 'AIPERF_SERVER_METRICS_URLS',
             'AIPERF_SERVER_URL'}
differences = {k: {'baseline': a.get(k), 'current': b.get(k)}
               for k in a.keys() | b.keys() if a.get(k) != b.get(k)}
errors = []
for k in set(differences) - path_keys:
    errors.append(f'Unexpected benchmark setting: {k}')
expected = {'CONC': '80', 'DURATION': '3600', 'AIPERF_WARMUP_REQUESTS_PER_LANE': '10',
            'SIMULATE_ACC_LEN': '3.61'}
for k, value in expected.items():
    if b.get(k) != value:
        errors.append(f'{k} != {value}')
dataset = Path(b['HF_HOME'])/'hub/datasets--semianalysisai--cc-traces-weka-062126'
revision = (dataset/'refs/main').read_text().strip()
if revision != '23f152f6f0f9399a85901b89a6458def0ef16729':
    errors.append(f'Dataset revision changed: {revision}')
manifest = {}
for p in sorted((dataset/'snapshots'/revision).rglob('*')):
    if p.is_file():
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(8*1024*1024), b''):
                h.update(block)
        manifest[str(p.relative_to(dataset/'snapshots'/revision))] = {'sha256': h.hexdigest(), 'bytes': p.stat().st_size}
if not manifest:
    errors.append('Empty dataset snapshot')
report = {'passed': not errors, 'errors': errors, 'runtime_differences': differences,
          'dataset_revision': revision, 'workload_manifest': manifest,
          'dataset_download_offline': os.environ.get('AGENTX_DATASET_DOWNLOAD_OFFLINE'),
          'dataset_pinner': os.environ.get('AGENTX_DATASET_PINNER')}
(out/'baseline-validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'passed': not errors, 'errors': errors, 'dataset_revision': revision}))
if errors:
    raise SystemExit(1)
