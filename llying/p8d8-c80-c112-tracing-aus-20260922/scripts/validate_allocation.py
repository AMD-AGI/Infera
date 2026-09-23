#!/usr/bin/env python3
"""Verify the configured allocation is still usable before starting load."""
import datetime
import json
import os
from pathlib import Path
import re
import subprocess

job = os.environ.get('ALLOCATION_JOB_ID')
if not job:
    raise SystemExit(0)
raw = subprocess.check_output(['scontrol', 'show', 'job', job], text=True, timeout=20)
fields = dict(re.findall(r'(\w+)=([^\s]+)', raw))
nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True, timeout=20).split()
expected = {os.environ['PREFILL_NODE'], os.environ['DECODE_NODE']}
errors = []
if fields.get('JobState') != 'RUNNING' or fields.get('PreemptTime') not in ('None', None):
    errors.append('Allocation is not running or is marked for preemption')
if set(nodes) != expected:
    errors.append(f'Allocation nodes changed: {nodes}')
if not fields.get('UserId', '').startswith('liyingli('):
    errors.append('Allocation owner mismatch')
report = {'checked_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'job_id': job, 'passed': not errors, 'errors': errors, 'slurm': raw}
with (Path(os.environ['RUN'])/'allocation-checks.jsonl').open('a') as f:
    f.write(json.dumps(report)+'\n')
print(json.dumps({k:v for k,v in report.items() if k != 'slurm'}))
if errors:
    raise SystemExit(1)
