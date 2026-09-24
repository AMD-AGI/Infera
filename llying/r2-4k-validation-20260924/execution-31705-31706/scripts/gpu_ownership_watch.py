#!/usr/bin/env python3
"""Invalidate a case if an unrelated process starts using its allocated GPUs."""
import concurrent.futures,json,os,shlex,signal,subprocess,time
from pathlib import Path
r=Path(os.environ['RUN']);prefix=os.environ['CONTAINER_PREFIX'];opts=shlex.split(os.environ['SSH_OPTS']);running=True
PROBE=r'''
from pathlib import Path
import json,subprocess,sys
s=subprocess.check_output(['docker','top',sys.argv[1],'-eo','pid'],text=True)
own={int(x.strip()) for x in s.splitlines() if x.strip().isdigit()}
foreign=[]
for p in Path('/sys/class/kfd/kfd/proc').iterdir():
 if not p.name.isdigit() or int(p.name) in own:continue
 try:
  used=sum(int(f.read_text().strip()) for f in p.glob('vram_*'))
  q=p/'queues';queues=len(list(q.iterdir())) if q.exists() else 0
 except FileNotFoundError:continue
 if used or queues:foreign.append({'pid':int(p.name),'vram_bytes':used,'queues':queues})
if foreign:
 s=subprocess.check_output(['docker','top',sys.argv[1],'-eo','pid'],text=True)
 own.update(int(x.strip()) for x in s.splitlines() if x.strip().isdigit())
 foreign=[x for x in foreign if x['pid'] not in own]
print(json.dumps({'owned_pids':sorted(own),'foreign':foreign}))
'''
def stop(*_):
 global running;running=False
signal.signal(signal.SIGTERM,stop)
def probe(role):
 node=os.environ[role.upper()+'_NODE'];cmd=['ssh',*opts,node,shlex.join(['python3','-',f'{prefix}-{role}-0'])]
 try:
  x=json.loads(subprocess.check_output(cmd,input=PROBE,text=True,timeout=20));return role,x
 except Exception as e:return role,{'error':str(e)}
failures=0
while running and not (r/'c80-completed.txt').exists():
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:rows=dict(pool.map(probe,['prefill','decode']))
 bad={role:[p for p in x.get('foreign',[]) if p['vram_bytes']>32*1024**2 or p['queues']>0] for role,x in rows.items()}
 failures=failures+1 if any('error' in x for x in rows.values()) else 0
 event={'time':time.time(),'roles':rows,'interference':any(bad.values()),'consecutive_probe_errors':failures}
 with (r/'sampling/gpu-ownership.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
 if event['interference']:
  (r/'INVALID').write_text(json.dumps(event,indent=2)+'\n')
  for _ in range(12):
   names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
   clients=[x for x in names if x.startswith(prefix+'-agentx-client-')]
   if clients:
    for name in clients:subprocess.run(['docker','stop','--timeout','10',name],check=False)
    break
   if not running:break
   time.sleep(5)
  raise SystemExit('Case invalidated: foreign GPU use')
 for _ in range(15):
  if not running:break
  time.sleep(1)
