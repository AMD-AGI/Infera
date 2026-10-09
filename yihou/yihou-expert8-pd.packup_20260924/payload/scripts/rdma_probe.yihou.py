#!/usr/bin/env python3
"""Run exact-image Mooncake GPU byte verification on four matched rails."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess

W = Path(__file__).resolve().parents[1]
IMAGE = 'infera-sglang:v0519-yihou-0917-nextnfix-hicache'
NODES = {138: '10.245.157.237', 136: '10.245.154.168'}

def remote(node, args, **kwargs):
    return subprocess.run(['ssh', '-o', 'ClearAllForwardings=yes', '-o', 'BatchMode=yes',
                           '-o', 'ConnectTimeout=10', f'crsuse2-m2m-{node}', shlex.join(args)],
                          **kwargs)

def ensure(node):
    name = f'yihou-expert8-rdma-ram-{node}'
    r = remote(node, ['docker', 'inspect', '--format', '{{.State.Running}}', name],
               capture_output=True, text=True)
    if r.returncode == 0:
        if r.stdout.strip() != 'true':
            remote(node, ['docker','start',name], check=True)
        return name
    if 'No such' not in r.stderr and 'no such' not in r.stderr:
        raise RuntimeError(r.stderr)
    cache, tmp = W / f'cache/leader-{node}', W / f'tmp/leader-{node}'
    cache.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    args = ['docker', 'run', '-d', '--name', name, '--network', 'host', '--ipc', 'host',
            '--device', '/dev/kfd', '--device', '/dev/dri', '--device', '/dev/infiniband',
            '--group-add', 'video', '--group-add', 'render', '--cap-add', 'IPC_LOCK',
            '--security-opt', 'seccomp=unconfined', '--ulimit', 'memlock=-1:-1',
            '-v', f'{W}:{W}', '-v', '/lib/x86_64-linux-gnu/libionic.so:/host-libionic/libionic.so:ro',
            '-w', str(W), '-e', 'HIP_VISIBLE_DEVICES=0,1,2,3', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '--tmpfs', f'{tmp}:rw,exec,size=8g',
            '-e', f'TMPDIR={tmp}', '-e', f'XDG_CACHE_HOME={cache}',
            '-e', f'TRITON_CACHE_DIR={cache}/triton', '-e', f'AITER_JIT_DIR={cache}/aiter',
            IMAGE, 'sleep', 'infinity']
    remote(node, args, check=True)
    return name

def endpoint(node, name, spec, logfile):
    args = ['docker', 'exec', '-e', 'MC_GID_INDEX=1', '-e', 'MC_ENABLE_DEST_DEVICE_AFFINITY=1',
            '-e', 'MC_DISABLE_HIP_TRANSPORT=1', '-e', 'MOONCAKE_DISABLE_HIP_DMABUF=0',
            '-e', 'RDMAV_FORK_SAFE=1', '-e', f'MC_TE_FILTERS={spec["dev"]}',
            name, 'timeout', '240', 'python3', '-X', 'faulthandler',
            str(W/'scripts/probe_worker.yihou.py'), json.dumps(spec)]
    logfile.with_suffix('.command.json').write_text(json.dumps({'node':node, 'args':args}, indent=2))
    with logfile.open('w') as log:
        return remote(node, args, stdout=log, stderr=subprocess.STDOUT, timeout=260).returncode

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(W):
        raise ValueError('output must be inside workspace')
    out.mkdir(parents=True, exist_ok=False)
    names = {n: ensure(n) for n in NODES}
    rows = []
    for source, target in [(138,136),(136,138)]:
        for gpu in range(4):
            sig = out / f'{source}-to-{target}-gpu{gpu}'
            sig.mkdir()
            base = dict(sig=str(sig), protocol='rdma', operation='write', loc='gpu',
                        gpu=gpu, dev=f'ionic_{gpu}')
            target_spec = dict(base, role='target', hostname=f'{NODES[target]}:0', host=f'crsuse2-m2m-{target}')
            initiator_spec = dict(base, role='initiator', hostname=f'{NODES[source]}:0', host=f'crsuse2-m2m-{source}')
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                ft = pool.submit(endpoint,target,names[target],target_spec,sig/'target.log')
                fi = pool.submit(endpoint,source,names[source],initiator_spec,sig/'initiator.log')
                trc, irc = ft.result(), fi.result()
            result = json.loads((sig/'result.json').read_text()) if (sig/'result.json').exists() else {}
            row = dict(result, source_node=source, target_node=target, target_rc=trc, initiator_rc=irc)
            rows.append(row)
            (out/'summary.json').write_text(json.dumps(rows,indent=2))
            print(json.dumps(row),flush=True)
            if trc or irc or result.get('verified') is not True or not result.get('gb_s'):
                raise RuntimeError(f'GPU RDMA validation failed: {row}')
    print('PASS: all eight directed GPU/rail byte checks',flush=True)

if __name__ == '__main__':
    main()
