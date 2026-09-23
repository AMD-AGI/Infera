#!/usr/bin/env python3
"""Wait for the preceding run to exit and release GPU and host cache memory."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

PROBE = r'''
from pathlib import Path
import json,subprocess,sys
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
 k,v=line.split(':',1)
 if k in ('MemTotal','MemAvailable','Shmem','SwapTotal','SwapFree'):
  mem[k]=int(v.split()[0])*1024
cards=[]
for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
 p=d/'mem_info_vram_used'
 if p.exists():
  cards.append({'card':d.parent.name,'used_bytes':int(p.read_text()),'total_bytes':int((d/'mem_info_vram_total').read_text()),'busy_pct':int((d/'gpu_busy_percent').read_text())})
c=subprocess.run(['docker','inspect',sys.argv[1],'--format','{{json .State}}'],capture_output=True,text=True)
if c.returncode:
 raise RuntimeError(c.stderr)
print(json.dumps({'mem':mem,'gpus':cards,'previous_container_state':json.loads(c.stdout)}))
'''
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--previous-prefix',required=True)
p.add_argument('--timeout',type=int,default=5400)
a=p.parse_args()
a.output.parent.mkdir(parents=True,exist_ok=True)
opts=shlex.split(os.environ['SSH_OPTS'])
roles={'prefill':os.environ['PREFILL_NODE'],'decode':os.environ['DECODE_NODE']}
def probe(item):
 role,node=item
 try:
  cmd=['ssh',*opts,node,shlex.join(['python3','-',f'{a.previous_prefix}-{role}-0'])]
  result=subprocess.run(cmd,input=PROBE,text=True,capture_output=True,timeout=30,check=True)
  data=json.loads(result.stdout)
  states=data['previous_container_state'];mem=data['mem'];cards=data['gpus']
  tests={'previous_container_exited':states['Status']=='exited' and states['Pid']==0,
         'gpu_idle':len(cards)==8 and all(c['used_bytes']/c['total_bytes']<.02 and c['busy_pct']<=5 for c in cards),
         'bulk_shared_memory_released':mem['Shmem']<64*1024**3,
         'host_memory_available':mem['MemAvailable']>(2*1024**4 if role=='prefill' else 512*1024**3)}
  return role,dict(node=node,passed=all(tests.values()),checks=tests,observed=data)
 except Exception as e:
  return role,dict(node=node,passed=False,error=str(e))
start=time.monotonic()
while True:
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  results=dict(pool.map(probe,roles.items()))
 row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'elapsed_seconds':round(time.monotonic()-start),'nodes':results,'passed':all(x['passed'] for x in results.values())}
 with a.output.open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps({'utc':row['utc'],'elapsed_seconds':row['elapsed_seconds'],'passed':row['passed'],'checks':{k:v.get('checks',v.get('error')) for k,v in results.items()}}),flush=True)
 if row['passed']:
  a.output.with_suffix('.passed.json').write_text(json.dumps(row,indent=2)+'\n')
  break
 if time.monotonic()-start>a.timeout:raise SystemExit('Resource release timeout: do not launch')
 time.sleep(15)
