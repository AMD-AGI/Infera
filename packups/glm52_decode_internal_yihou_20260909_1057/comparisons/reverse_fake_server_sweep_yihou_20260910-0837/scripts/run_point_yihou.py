#!/usr/bin/env python3
"""Run one guarded, evidence-preserving fake decode-server replay point."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

W = Path(__file__).resolve().parents[1]
REPO = W.parents[2]
IMAGE = 'sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d'
MODEL = '/shared_nfs/models/GLM-5.2-MXFP4'
JOB = '131080'
NODE = 'crsuse2-m2m-271'
PORT = 31916
PROBE = 'yihou-glm52-compare-20260910'


def remote(args, **kwargs):
    return subprocess.run(['spur', 'exec', JOB, *map(str, args)], **kwargs)


def capture(args, timeout=60):
    return remote(args, capture_output=True, text=True, timeout=timeout, check=True).stdout


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def commands(mode, concurrency, name, result):
    local_batch = concurrency // 4 if mode == 'on' else concurrency
    server = ['python3', '-m', 'sglang.launch_server',
        '--model-path', MODEL, '--served-model-name', 'glm52', '--trust-remote-code',
        '--host', '0.0.0.0', '--port', str(PORT), '--tp', '4', '--ep-size', '4',
        '--kv-cache-dtype', 'fp8_e4m3', '--dsa-prefill-backend', 'flydsl',
        '--dsa-decode-backend', 'flydsl', '--tool-call-parser', 'glm47',
        '--reasoning-parser', 'glm45', '--chunked-prefill-size', '32768',
        '--mem-fraction-static', '0.85', '--max-running-requests', str(concurrency),
        '--cuda-graph-max-bs', str(local_batch), '--speculative-algorithm', 'EAGLE',
        '--speculative-num-steps', '5', '--speculative-eagle-topk', '1',
        '--speculative-num-draft-tokens', '6', '--dsa-topk-backend', 'aiter',
        '--enable-aiter-allreduce-fusion', '--enable-fused-qk-norm-rope',
        '--watchdog-timeout', '1800', '--enable-metrics', '--decode-log-interval', '10',
        '--disaggregation-mode', 'decode', '--disaggregation-transfer-backend', 'fake']
    if mode == 'on':
        server += ['--dp-size', '4', '--enable-dp-attention', '--load-balance-method', 'round_robin']
    docker = ['docker', 'run', '-d', '--name', name, '--label', 'owner=yihou',
        '--label', f'experiment={W.name}', '-w', '/', '--network', 'host', '--ipc=host',
        '--shm-size', '32g', '--device=/dev/kfd', '--device=/dev/dri',
        '--group-add', 'video', '--group-add', 'render',
        '--security-opt', 'seccomp=unconfined', '--ulimit', 'memlock=-1:-1',
        '-v', f'{W}:{W}', '-v', f'{MODEL}:{MODEL}:ro']
    for source, destination in [
        ('dsa_utils_fake_fix.py', 'layers/attention/dsa/utils.py'),
        ('kv_cache_configurator_fake_fix.py', 'mem_cache/kv_cache_configurator.py'),
        ('eagle_disaggregation_fake_fix.py', 'speculative/eagle_disaggregation.py')]:
        docker += ['-v', f'{W}/source/{source}:/sglang/python/sglang/srt/{destination}:ro']
    for directory, destination, variable in [
        ('jit-b9a83742f631', '/jit-cache', 'AITER_JIT_DIR'),
        ('triton', '/triton-cache', 'TRITON_CACHE_DIR'),
        ('torch', '/torch-cache', 'TORCHINDUCTOR_CACHE_DIR'),
        ('hf', '/hf_cache', 'HF_HOME')]:
        (W / 'cache' / directory).mkdir(parents=True, exist_ok=True)
        docker += ['-v', f'{W}/cache/{directory}:{destination}', '-e', f'{variable}={destination}']
    env = {'SGLANG_OPT_USE_TOPK_V2': 'false', 'SGLANG_TIMEOUT_KEEP_ALIVE': '900',
        'HIP_VISIBLE_DEVICES': '0,1,2,3', 'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
        'AITER_USE_FLYDSL_MOE_SORTING': '1', 'SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK': '1',
        'SGLANG_SIMULATE_ACC_LEN': '3.61', 'SGLANG_SIMULATE_ACC_METHOD': 'match-expected',
        'SGLANG_SIMULATE_ACC_TOKEN_MODE': 'real-draft-token'}
    for key, value in env.items():
        docker += ['-e', f'{key}={value}']
    client = ['docker', 'exec', '-e', 'PYTHONPATH=/sglang/python', '-e', 'PYTHONDONTWRITEBYTECODE=1',
        name, 'python3', '-m', 'sglang.benchmark.serving', '--backend', 'sglang',
        '--host', '127.0.0.1', '--port', str(PORT), '--model', MODEL,
        '--dataset-name', 'random', '--dataset-path', str(W / 'reference/synthetic_prompts.json'),
        '--tokenize-prompt', '--random-input-len', '70000', '--random-output-len', '10000',
        '--random-range-ratio', '1', '--num-prompts', '128', '--max-concurrency', str(concurrency),
        '--warmup-requests', '16', '--fake-prefill', '--output-details',
        '--output-file', str(result / 'benchmark.jsonl')]
    return docker + [IMAGE] + server, client, server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['off', 'on'])
    p.add_argument('concurrency', type=int, choices=[4, 8, 16, 20, 24])
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    point = f'dpa_{a.mode}_c{a.concurrency}_yihou'
    name = f'yihou-glm52-reverse-0910-{a.mode}-c{a.concurrency}'
    result = W / 'rounds' / point
    launch, client, server = commands(a.mode, a.concurrency, name, result)
    if a.dry_run:
        print(shlex.join(launch)); print(shlex.join(client)); return
    if result.exists():
        raise SystemExit(f'Refusing to overwrite {result}')
    queue = subprocess.check_output(['squeue', '-j', JOB, '-h', '-o', '%i %u %T %N'], text=True).split()
    if queue != [JOB, 'yihou', 'RUNNING', NODE]:
        raise SystemExit(f'Allocation mismatch: {queue}')
    if capture(['hostname']).strip() != NODE:
        raise SystemExit('Wrong node')
    existing = remote(['docker', 'inspect', name], capture_output=True, text=True, timeout=30)
    if existing.returncode == 0:
        raise SystemExit('Refusing to reuse existing container')
    gpu = capture(['docker', 'exec', '-w', '/', PROBE, 'rocm-smi', '--showmemuse', '--showpids'])
    vram = [int(v) for v in re.findall(r'GPU Memory Allocated \(VRAM%\):\s*(\d+)', gpu)]
    if len(vram) != 8 or max(vram) > 2:
        raise SystemExit(f'GPU idle gate failed: {vram}')
    listening = remote(['curl', '-sS', '--max-time', '3', f'http://127.0.0.1:{PORT}/server_info'], capture_output=True, timeout=10)
    if listening.returncode == 0:
        raise SystemExit('Replay port already serves HTTP')
    result.mkdir()
    (result / 'gpu-before.txt').write_text(gpu)
    (result / 'server-command.txt').write_text(shlex.join(server) + '\n')
    (result / 'docker-command.txt').write_text(shlex.join(launch) + '\n')
    (result / 'benchmark-command.txt').write_text(shlex.join(client) + '\n')
    (result / 'runner_snapshot_yihou.py').write_bytes(Path(__file__).read_bytes())
    (result / 'git-head.txt').write_text(subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True))
    (result / 'code.diff').write_bytes(subprocess.check_output(['git', '-C', str(REPO), 'diff', '--', str(W)]))
    save(result / 'source_hashes.json', {str(f.relative_to(W)): hashlib.sha256(f.read_bytes()).hexdigest()
        for folder in ['scripts', 'source', 'reference'] for f in (W / folder).rglob('*') if f.is_file()})
    status: dict[str, object] = dict(point=point, container=name, job=JOB, node=NODE,
        started_at=datetime.now(timezone.utc).isoformat(), state='starting')
    start = time.monotonic()
    save(result / 'status.json', status)
    launched = False
    try:
        container_id = capture(launch, timeout=120).strip()
        launched = True
        (result / 'container-id.txt').write_text(container_id + '\n')
        (result / 'container-inspect.json').write_text(capture(['docker', 'inspect', name]))
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            r = remote(['curl', '-fsS', '--max-time', '5', f'http://127.0.0.1:{PORT}/server_info'],
                capture_output=True, text=True, timeout=25)
            if r.returncode == 0:
                info = json.loads(r.stdout)
                save(result / 'server-info-before.json', info)
                break
            running = capture(['docker', 'inspect', name, '--format', '{{.State.Running}}']).strip()
            if running != 'true':
                raise RuntimeError('Server exited before readiness')
            status['startup_elapsed_seconds'] = round(time.monotonic() - start)
            save(result / 'status.json', status)
            print(json.dumps(status), flush=True)
            time.sleep(20)
        else:
            raise TimeoutError('Readiness timeout; server retained for diagnosis')
        status.update(state='benchmarking', startup_seconds=time.monotonic() - start)
        save(result / 'status.json', status)
        print(json.dumps(status), flush=True)
        bench_start = time.monotonic()
        with (result / 'benchmark.log').open('w') as log:
            run = remote(client, stdout=log, stderr=subprocess.STDOUT, timeout=5400)
        status.update(client_exit=run.returncode, client_wall_seconds=time.monotonic() - bench_start)
        if run.returncode:
            raise RuntimeError(f'Client failed: {run.returncode}')
        save(result / 'server-info-after.json', json.loads(capture(['curl', '-fsS', '--max-time', '15', f'http://127.0.0.1:{PORT}/server_info'])))
        (result / 'metrics-after.txt').write_text(capture(['curl', '-fsS', '--max-time', '15', f'http://127.0.0.1:{PORT}/metrics']))
        records = [json.loads(line) for line in (result / 'benchmark.jsonl').read_text().splitlines() if line.strip()]
        measured = records[-1]
        if measured.get('completed') != 128 or measured.get('total_output_tokens') != 1280000:
            raise RuntimeError(f'Unexpected request/token counts: {measured.get("completed")}, {measured.get("total_output_tokens")}')
        from collect_yihou import validate
        save(result / 'verification_yihou.json', validate(measured, a.mode, a.concurrency))
        status['state'] = 'passed'
        # Exact owned server only. Keep the stopped container and every artifact.
        identity = json.loads(capture(['docker', 'inspect', name]))[0]
        if identity['Id'] != container_id or identity['Config']['Labels'].get('experiment') != W.name:
            raise RuntimeError('Container ownership changed; refusing stop')
        capture(['docker', 'stop', '--time', '30', name], timeout=90)
    except Exception as exc:
        status.update(state='failed', error=str(exc))
        raise
    finally:
        if launched:
            with (result / 'server.log').open('w') as log:
                remote(['docker', 'logs', '--timestamps', name], stdout=log, stderr=subprocess.STDOUT, timeout=120)
        status['wall_seconds'] = time.monotonic() - start
        status['finished_at'] = datetime.now(timezone.utc).isoformat()
        save(result / 'status.json', status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
