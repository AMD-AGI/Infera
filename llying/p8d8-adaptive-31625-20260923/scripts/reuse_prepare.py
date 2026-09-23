#!/usr/bin/env python3
"""Prepare an idle P/D pair for a fresh, independently captured reuse trial."""
import json,os,shlex,subprocess,time,urllib.request
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from sample_engine_metrics import parse_metrics
r=Path(os.environ['RUN']);root=Path(os.environ['TRACE_RUNTIME']);prefix=os.environ['CONTAINER_PREFIX']
opts=shlex.split(os.environ['SSH_OPTS'])
roles={'prefill':(os.environ['PREFILL_NODE'],os.environ['PREFILL_IP'],29001),'decode':(os.environ['DECODE_NODE'],os.environ['DECODE_IP'],29002)}
def ssh(node,args,input=None):
 return subprocess.check_output(['ssh',*opts,node,shlex.join(args)],input=input,text=True,timeout=120)
def metrics(ip,port):
 with urllib.request.urlopen(f'http://{ip}:{port}/metrics',timeout=20) as f:return parse_metrics(f.read().decode())[0]
def get_json(url):
 with urllib.request.urlopen(url,timeout=30) as f:return json.load(f)
def inspect(node,name):return json.loads(ssh(node,['docker','inspect',name]))[0]
queue_names={'sglang:num_queue_reqs','sglang:num_running_reqs','sglang:num_decode_prealloc_queue_reqs','sglang:num_decode_transfer_queue_reqs','sglang:num_prefill_bootstrap_queue_reqs','sglang:num_prefill_inflight_queue_reqs','sglang:num_prefill_prealloc_queue_reqs'}
r.mkdir(parents=True,exist_ok=True);(r/'snapshot').mkdir(exist_ok=True)
before={};report={'roles':{},'started_at':time.time()}
for role,(node,ip,port) in roles.items():
 rows=metrics(ip,port);busy=[x for x in rows if x['metric'] in queue_names and isinstance(x['value'],(int,float)) and x['value']!=0]
 if busy:raise SystemExit(f'{role}: in-flight requests present: {busy}')
 before[role]=inspect(node,f'{prefix}-{role}-0')
 report['roles'][role]={'container_id':before[role]['Id'],'pid':before[role]['State']['Pid'],'before_metrics':rows,'server_info':get_json(f'http://{ip}:{port}/get_server_info')}
(r/'snapshot/reuse-before.json').write_text(json.dumps(report,indent=2)+'\n')
# Stop only the old control plane. Engines remain alive.
for suffix in ('router','collector'):
 name=f'{prefix}-{suffix}'
 c=inspect(os.environ['PREFILL_NODE'],name)
 (r/f'snapshot/previous-{suffix}.json').write_text(json.dumps(c,indent=2)+'\n')
 subprocess.run(['docker','stop','--timeout','30',name],check=True)
 subprocess.run(['docker','rename',name,f'{name}-before-{os.environ["RUN_ID"]}'],check=True)
for role,(node,ip,port) in roles.items():
 req=urllib.request.Request(f'http://{ip}:{port}/flush_cache',data=b'',method='POST')
 with urllib.request.urlopen(req,timeout=180) as f:body=f.read().decode()
 (r/f'snapshot/flush-{role}.txt').write_text(body)
# Metrics are emitted periodically. Require all 8 rank series and empty logical pools.
for attempt in range(30):
 good=True
 for role,(node,ip,port) in roles.items():
  rows=metrics(ip,port)
  required={'sglang:num_running_reqs','sglang:num_queue_reqs','sglang:kv_used_tokens','sglang:kv_evictable_tokens'}
  if role=='prefill':required.add('sglang:hicache_host_used_tokens')
  checks={}
  for metric in required:
   selected=[x for x in rows if x['metric']==metric]
   ranks={x.get('labels',{}).get('dp_rank') for x in selected}
   checks[metric]=len(ranks)==8 and all(isinstance(x['value'],(int,float)) and 0<=x['value']<=64 for x in selected)
  checks['queues_empty']=all(x['value']==0 for x in rows if x['metric'] in queue_names)
  after=inspect(node,f'{prefix}-{role}-0')
  checks['engine_identity_unchanged']=after['Id']==before[role]['Id'] and after['State']['Pid']==before[role]['State']['Pid'] and after['State']['Running']
  report['roles'][role].update(after_metrics=rows,checks=checks)
  good &= all(checks.values())
 report['passed']=bool(good);report['checked_at']=time.time()
 (r/'snapshot/reuse-reset-validation.json').write_text(json.dumps(report,indent=2)+'\n')
 if good:break
 time.sleep(2)
else:raise SystemExit('Flush did not restore empty logical GPU/host caches on all ranks')
# Save byte cursors so subsequent captures exclude earlier experiments.
cursors={}
probe='''from pathlib import Path
import json,sys
p=Path(sys.argv[1]);print(json.dumps({x.name:{"size":x.stat().st_size,"inode":x.stat().st_ino} for x in p.glob("*.jsonl")}))'''
for role,(node,ip,port) in roles.items():
 srcid=os.environ.get(f'DIAG_SOURCE_{role.upper()}',os.environ['SERVICE_RUN_ID'])
 cursors[role]=json.loads(ssh(node,['python3','-c',probe,f'/tmp/aus-diag-{srcid}/{role}']))
(r/'snapshot/diagnostic-cursors.json').write_text(json.dumps(cursors,indent=2)+'\n')
(r/'snapshot/capture-start-epoch.txt').write_text(str(time.time())+'\n')
print(json.dumps({'passed':True,'run':str(r),'engine_reuse':True}),flush=True)
