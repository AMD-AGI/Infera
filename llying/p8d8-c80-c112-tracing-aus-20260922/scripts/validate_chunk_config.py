#!/usr/bin/env python3
"""Fail closed before smoke/load if the configured chunk run deviates from baseline."""
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess

run = Path(os.environ['RUN'])
baseline = Path(os.environ['BASELINE_RUN'])
root = Path(os.environ['TRACE_RUNTIME'])
report = {'baseline': str(baseline), 'roles': {}, 'errors': []}
errors = report['errors']
old_containers = json.loads((baseline/'live-containers.json').read_text())
expected_image = 'sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
replacements = {
    '10.235.192.136': os.environ['PREFILL_IP'],
    '10.235.192.128': os.environ['DECODE_IP'],
}
def normalize(value):
    if isinstance(value, str):
        for old, new in replacements.items():
            value = value.replace(old, new)
    return value

def diffs(old, new):
    return {key: {'baseline': old.get(key), 'current': new.get(key)}
            for key in old.keys() | new.keys() if normalize(old.get(key)) != new.get(key)}

# Runtime/process identifiers and memory-derived capacity are recorded separately.
runtime_keys = {'status', 'startup_time', 'scheduler_pids', 'internal_states',
                'launch_command', 'nccl_port', 'random_seed', 'max_total_num_tokens',
                'max_req_input_len', 'kv_events'}
for role in ('prefill', 'decode'):
    old = json.loads((baseline/f'launch/server-info/{role}-0.json').read_text())
    new = json.loads((run/f'launch/server-info/{role}-0.json').read_text())
    differences = diffs(old, new)
    expected_chunk = int(os.environ['EXPECTED_PREFILL_CHUNK']) if role == 'prefill' else 4096
    if new.get('chunked_prefill_size') != expected_chunk:
        errors.append(f'{role}: effective chunk is not {expected_chunk}')
    dynamic_fields = set()
    if 'kv_events_config' in differences:
        old_events, new_events = json.loads(old['kv_events_config']), json.loads(new['kv_events_config'])
        endpoints = [old_events.pop('endpoint', ''), new_events.pop('endpoint', '')]
        if old_events == new_events and all(re.fullmatch(r'tcp://\*:[0-9]+', endpoint) for endpoint in endpoints):
            dynamic_fields.add('kv_events_config')
    unexpected = set(differences) - runtime_keys - dynamic_fields - ({'chunked_prefill_size'} if role == 'prefill' else set())
    if unexpected:
        errors.append(f'{role}: unexpected server fields {sorted(unexpected)}')
    # Large capacity differences require explicit review before any workload.
    a, b = old.get('max_total_num_tokens'), new.get('max_total_num_tokens')
    if not a or not b or abs(b/a-1) > .01:
        errors.append(f'{role}: KV capacity differs by more than 1%: {a} -> {b}')
    node = os.environ[f'{role.upper()}_NODE']
    container = json.loads(subprocess.check_output(
        ['ssh', *shlex.split(os.environ['SSH_OPTS']), node, 'docker', 'inspect',
         f"{os.environ['CONTAINER_PREFIX']}-{role}-0"], text=True))[0]
    previous = old_containers[role]
    if container['Image'] != expected_image or previous['Image'] != expected_image:
        errors.append(f'{role}: image mismatch')
    old_cmd = [normalize(x) for x in previous['Config']['Cmd']]
    if role == 'prefill':
        old_cmd[old_cmd.index('--chunked-prefill-size')+1] = os.environ['PREFILL_CHUNK_SIZE']
    if old_cmd != container['Config']['Cmd']:
        errors.append(f'{role}: unexpected container command')
    env = lambda c: dict(x.split('=', 1) for x in c['Config']['Env'] if '=' in x)
    env_diff = diffs(env(previous), env(container))
    if env_diff:
        errors.append(f'{role}: unexpected environment {sorted(env_diff)}')
    report['roles'][role] = {'server_differences': differences, 'environment_differences': env_diff,
                             'container': container}
helper = root/'docker/aus_diag.py'
old_helper = baseline/'live-diag-helper.py'
report['helper_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
if helper.read_bytes() != old_helper.read_bytes():
    errors.append('Diagnostic helper differs from baseline')
patch = root/'scripts/patch_client.py'
if patch.read_bytes() != (baseline/'c80/executed-client-patch.py').read_bytes():
    errors.append('Client patch differs from baseline')
repo = Path(os.environ['INFERENCEX_DIR'])
commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
report['inferencex_commit'] = commit
if commit != '918524ff94045b3f091115f1051c22a8588edf2b':
    errors.append('InferenceX commit differs from baseline')
if subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True).strip():
    errors.append('InferenceX checkout is dirty')
report['passed'] = not errors
(run/'chunk-config-validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'passed': not errors, 'errors': errors, 'report': str(run/'chunk-config-validation.json')}))
if errors:
    raise SystemExit(1)
