#!/usr/bin/env python3
"""Keep cleanup independent of the SSH session and job-step lifetime."""
import argparse,concurrent.futures,datetime,http.client,json,signal,socket,time
from pathlib import Path
from allocation_state import snapshot
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--role',required=True);p.add_argument('--once',action='store_true');a=p.parse_args()
prefix='llying-r2-31705-31706';image='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb';events=a.root/'events'/('container-guard-'+a.role+'.jsonl');stop=False
class Docker(http.client.HTTPConnection):
 def __init__(self):super().__init__('localhost',timeout=45)
 def connect(self):
  self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);self.sock.settimeout(45);self.sock.connect('/var/run/docker.sock')
def api(method,path):
 c=Docker()
 try:
  c.request(method,path);r=c.getresponse();body=r.read()
  if r.status not in (200,204,304):raise RuntimeError(f'Docker {method} {path}: {r.status} {body[:200]!r}')
  return json.loads(body) if body else None
 finally:c.close()
def record(data):
 data['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with events.open('a') as f:f.write(json.dumps(data)+'\n')
def cleanup(reason):
 record({'event':'cleanup_requested','reason':reason})
 chosen=[]
 allowed={'prefill-0','router','collector','etcd'} if a.role=='prefill' else {'decode-0'}
 for c in api('GET','/containers/json'):
  name=c['Names'][0].lstrip('/')
  if not name.startswith(prefix+'-') or (name[len(prefix)+1:] not in allowed and not name[len(prefix)+1:].startswith('agentx-client-')):continue
  d=api('GET','/containers/'+c['Id']+'/json')
  if name.endswith('-etcd'):
   if d['Config']['Image']!='quay.io/coreos/etcd:v3.5.14':continue
  elif d['Image']!=image:continue
  chosen.append((name,c['Id']))
 def end(item):
  name,cid=item
  try:api('POST','/containers/'+cid+'/stop?t=5');record({'event':'stopped','name':name,'id':cid})
  except Exception as e:record({'event':'stop_error','name':name,'id':cid,'error':str(e)})
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(end,chosen))
 run=a.root/'runs/r2-on-4k-31705-31706'
 if not (run/'c80-completed.txt').exists() and run.exists():(run/'INVALID').write_text(reason+'\n')
def requested(*_):
 global stop;stop=True
signal.signal(signal.SIGTERM,requested);signal.signal(signal.SIGINT,requested)
errors=0
while True:
 if stop:cleanup('guard termination');break
 if (a.root/'events/cleanup-complete').exists():break
 try:
  state=snapshot(a.root);errors=0
  record({'event':'heartbeat','passed':state['passed'],'errors':state['errors']})
  if not state['passed']:
   if not a.once:cleanup('; '.join(state['errors']))
   break
 except Exception as e:
  errors+=1;record({'event':'query_error','error':str(e),'consecutive':errors})
  if errors>=3:
   if not a.once:cleanup('allocation status unavailable')
   break
 if a.once:break
 time.sleep(5)
