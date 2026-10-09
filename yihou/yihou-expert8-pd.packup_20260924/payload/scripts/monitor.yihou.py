#!/usr/bin/env python3
"""Capture bounded live container logs and engine health during bring-up/load."""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

W=Path(__file__).resolve().parents[1]
TARGETS=[('prefill',138,'10.245.157.237',29001),('decode',136,'10.245.154.168',29002)]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seconds',type=int,default=3600)
    args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(W):raise ValueError('output outside workspace')
    out.mkdir(parents=True,exist_ok=True)
    end=time.monotonic()+args.seconds
    while time.monotonic()<end and not (out/'stop.yihou').exists():
        rows=[]
        for role,node,ip,port in TARGETS:
            name=f'glm52-pd-yihou-expert8-{role}-0'
            command=f'docker inspect --format "{{{{json .State}}}}" {name}; docker logs --since 70s --timestamps {name}'
            try:
                r=subprocess.run(['ssh','-o','ClearAllForwardings=yes','-o','BatchMode=yes',
                    '-o','ConnectTimeout=8',f'crsuse2-m2m-{node}',command],capture_output=True,text=True,timeout=15)
                with (out/f'{role}.live.log').open('a') as log:
                    log.write(f'\n=== {time.time()} ===\n'+r.stdout+r.stderr)
                row={'role':role,'inspect_rc':r.returncode}
                try:
                    with urllib.request.urlopen(f'http://{ip}:{port}/health',timeout=5) as response:
                        row['health']=response.status
                except Exception as exc:row['health_error']=str(exc)
                errors=[term for term in ('Segmentation fault','Memory access fault','Traceback','NCCL error') if term in r.stdout+r.stderr]
                row['error_markers']=errors
                rows.append(row)
            except Exception as exc:rows.append({'role':role,'error':repr(exc)})
        with (out/'health.jsonl').open('a') as log:
            log.write(json.dumps({'time':time.time(),'rows':rows})+'\n')
        print(json.dumps(rows),flush=True)
        time.sleep(60)

if __name__=='__main__':main()
