#!/usr/bin/env python3
"""Snapshot exact live deployment evidence without relying on log followers."""
import argparse
import concurrent.futures
import json
import re
import shlex
import subprocess
from pathlib import Path

W=Path(__file__).resolve().parents[1]
TARGETS=[('prefill',138,'10.245.157.237',29001),('decode',136,'10.245.154.168',29002)]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(W):raise ValueError('output outside workspace')
    out.mkdir(parents=True,exist_ok=False)
    def capture(target):
        role,node,ip,port=target
        name=f'glm52-pd-yihou-expert8-{role}-0'
        commands={'inspect':['docker','inspect',name],
                  'engine.log':['docker','logs','--timestamps',name],
                  'gpu':['rocm-smi','--showpids','--showmeminfo','vram'],
                  'server-info':['curl','-fsS','--max-time','30',f'http://{ip}:{port}/get_server_info'],
                  'metrics':['curl','-fsS','--max-time','30',f'http://{ip}:{port}/metrics']}
        status={}
        for key,command in commands.items():
            try:
                result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ClearAllForwardings=yes',
                    '-o','ConnectTimeout=10',f'crsuse2-m2m-{node}',shlex.join(command)],
                    capture_output=True,text=True,timeout=45)
                (out/f'{role}-{key}.txt').write_text(result.stdout+result.stderr)
                status[key]=result.returncode
                if key=='engine.log':
                    lines=[line for line in result.stdout.splitlines() if re.search(
                        r'Load weight end|KV Cache is allocated|max_total_num_tokens|Memory pool end|Capture cuda graph end|yihou|expert8',line)]
                    (out/f'{role}-memory.txt').write_text('\n'.join(lines)+'\n')
            except Exception as exc:status[key]=repr(exc)
        return role,status
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        status=dict(pool.map(capture,TARGETS))
    (out/'status.json').write_text(json.dumps(status,indent=2))
    print(json.dumps(status,indent=2))

if __name__=='__main__':main()
