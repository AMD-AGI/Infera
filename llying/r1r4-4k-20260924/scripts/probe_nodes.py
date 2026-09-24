#!/usr/bin/env python3
"""Observe allocated node VRAM once; no launch or cleanup actions."""
import concurrent.futures,datetime,json,subprocess
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924')
assigned=json.loads((root/'config/assigned-nodes.json').read_text())
probe='''
from pathlib import Path
import json
rows=[]
for d in sorted(Path('/sys/class/drm').glob('card[0-9]*/device')):
 p=d/'mem_info_vram_used'
 if p.exists():
  rows.append({'card':d.parent.name,'vram_pct':round(100*int(p.read_text())/int((d/'mem_info_vram_total').read_text()),3),'busy_pct':int((d/'gpu_busy_percent').read_text())})
print(json.dumps(rows))
'''
def read(role):
 n=assigned[role]['node']
 p=subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','UserKnownHostsFile='+str(root/'config/known_hosts'),n,'python3','-'],input=probe,capture_output=True,text=True,timeout=25)
 return role,{'node':n,'gpus':json.loads(p.stdout)} if p.returncode==0 else {'node':n,'error':p.stderr.strip()}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:result=dict(pool.map(read,['prefill','decode']))
row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'roles':result}
with (root/'events/resource-readiness.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))
