#!/usr/bin/env python3
"""Fail closed before smoke/load if the configured chunk run deviates from baseline."""
import hashlib
import json
import os
import re
import urllib.request
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
replacements = {}
for role in ('prefill','decode'):
    old_env = dict(x.split('=',1) for x in old_containers[role]['Config']['Env'] if '=' in x)
    replacements[old_env['HOST_IP']] = os.environ[f'{role.upper()}_IP']
pattern = re.compile('|'.join(re.escape(x) for x in sorted(replacements,key=len,reverse=True)))
def normalize(value):
    return pattern.sub(lambda m: replacements[m[0]], value) if isinstance(value,str) else value

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
    permitted = {'chunked_prefill_size', 'mem_fraction_static', 'random_seed'}
    if role == 'prefill': permitted.update({'hicache_ratio','max_total_tokens'})
    unexpected = set(differences) - runtime_keys - dynamic_fields - permitted
    if new.get('mem_fraction_static') != float(os.environ[f'{role.upper()}_MEM_FRACTION']): errors.append(f'{role}: wrong memory fraction')
    if new.get('random_seed') != int(os.environ[f'{role.upper()}_SEED']): errors.append(f'{role}: wrong seed')
    if role == 'prefill' and new.get('hicache_ratio') != float(os.environ['PREFILL_HICACHE_RATIO']): errors.append('prefill: wrong host ratio')
    cap=int(os.environ.get('PREFILL_MAX_TOTAL_TOKENS','0')) if role=='prefill' else 0
    if new.get('max_total_tokens') != (cap or None):errors.append(f'{role}: unexpected explicit capacity limit')
    if unexpected:
        errors.append(f'{role}: unexpected server fields {sorted(unexpected)}')
    # Large capacity differences require explicit review before any workload.
    a, b = old.get('max_total_num_tokens'), new.get('max_total_num_tokens')
    expected_capacity = int(os.environ.get(f'EXPECTED_{role.upper()}_TOKENS','0'))
    if not b or (expected_capacity and b != expected_capacity):
        errors.append(f'{role}: unexpected KV capacity {b}; expected {expected_capacity}')
    states=new.get('internal_states',[])
    if len(states)!=8 or any(v.get('memory_usage',{}).get('token_capacity')!=b for v in states):errors.append(f'{role}: per-rank capacities disagree')
    node = os.environ[f'{role.upper()}_NODE']
    container = json.loads(subprocess.check_output(
        ['ssh', *shlex.split(os.environ['SSH_OPTS']), node, 'docker', 'inspect',
         f"{os.environ['CONTAINER_PREFIX']}-{role}-0"], text=True))[0]
    previous = old_containers[role]
    if container['Image'] != expected_image or previous['Image'] != expected_image:
        errors.append(f'{role}: image mismatch')
    old_cmd = [normalize(x) for x in previous['Config']['Cmd']]
    for flag, value in {'--chunked-prefill-size': os.environ[f'{role.upper()}_CHUNK_SIZE'], '--mem-fraction-static': os.environ[f'{role.upper()}_MEM_FRACTION']}.items():
        old_cmd[old_cmd.index(flag)+1] = value
    if role == 'prefill':old_cmd[old_cmd.index('--hicache-ratio')+1] = os.environ['PREFILL_HICACHE_RATIO']
    if '--random-seed' in old_cmd:old_cmd[old_cmd.index('--random-seed')+1] = os.environ[f'{role.upper()}_SEED']
    else:old_cmd += ['--random-seed',os.environ[f'{role.upper()}_SEED']]
    if cap:
        if '--max-total-tokens' in old_cmd:old_cmd[old_cmd.index('--max-total-tokens')+1]=str(cap)
        else:old_cmd += ['--max-total-tokens',str(cap)]
    if old_cmd != container['Config']['Cmd']:
        errors.append(f'{role}: unexpected container command')
    env = lambda c: dict(x.split('=', 1) for x in c['Config']['Env'] if '=' in x)
    env_diff = diffs(env(previous), env(container))
    if env_diff:
        errors.append(f'{role}: unexpected environment {sorted(env_diff)}')
    report['roles'][role] = {'server_differences': differences, 'environment_differences': env_diff,
                             'container': container}
# Check actual per-rank host capacity; ratio alone is not enough.
from sample_engine_metrics import parse_metrics
with urllib.request.urlopen(f"http://{os.environ['PREFILL_IP']}:29001/metrics",timeout=30) as response:
    series=parse_metrics(response.read().decode())[0]
host=[x for x in series if x['metric']=='sglang:hicache_host_total_tokens']
expected_host=int(os.environ.get('EXPECTED_HOST_TOKENS','0'))
report['host_capacity_samples']=host
if len({x['labels'].get('dp_rank') for x in host})!=8:errors.append('Host capacity does not cover eight ranks')
if expected_host and any(x['value']!=expected_host for x in host):errors.append('Host absolute token capacity changed')
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
(run/'case-validation.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'passed': not errors, 'errors': errors, 'report': str(run/'case-validation.json')}))
if errors:
    raise SystemExit(1)
