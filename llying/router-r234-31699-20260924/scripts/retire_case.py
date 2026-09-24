#!/usr/bin/env python3
"""Stop only this allocation's captured containers, then wait for resource release."""
import concurrent.futures,datetime,json,os,shlex,subprocess,time
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN']);prefix=os.environ['CONTAINER_PREFIX']
assert prefix=='llying-adaptive-31699'
phase=(run/'STATUS').read_text()
assert 'COMPLETE_REVIEW_PENDING' in phase or 'SMOKE_COMPLETE' in phase,phase
subprocess.run(['python3',str(root/'scripts/validate_allocation.py')],check=True)
if 'COMPLETE_REVIEW_PENDING' in phase:assert json.load(open(run/'analysis/case-audit.json'))['passed']
opts=shlex.split(os.environ['SSH_OPTS'])
def ssh(node,args,timeout=150):return subprocess.check_output(['ssh',*opts,node,shlex.join(args)],text=True,timeout=timeout)
prefill=os.environ['PREFILL_NODE'];decode=os.environ['DECODE_NODE']
ids=json.load(open(run/'live-containers.json'));rid=json.load(open(run/'snapshot/router-mode-validation.json'))['container']['Id']
expected='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
plan=[]
for suffix,node in [('router',prefill),('prefill-0',prefill),('decode-0',decode),('collector',prefill),('etcd',prefill)]:
 c=json.loads(ssh(node,['docker','inspect',prefix+'-'+suffix]))[0]
 assert c['Name']=='/'+prefix+'-'+suffix
 if suffix!='etcd':assert c['Image']==expected
 else:assert c['Config']['Image']=='quay.io/coreos/etcd:v3.5.14'
 if suffix in ('prefill-0','decode-0'):assert c['Id']==ids[suffix.split('-')[0]]['Id']
 if suffix=='router':assert c['Id']==rid
 plan.append({'node':node,'suffix':suffix,'id':c['Id'],'image':c['Image']})
for node in (prefill,decode):
 assert not any(n.startswith(prefix+'-agentx-client-') for n in ssh(node,['docker','ps','--format','{{.Names}}']).splitlines())
out=root/'events'/('retire-'+run.name);out.mkdir(exist_ok=True)
(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
def stop(c):
 ssh(c['node'],['docker','stop','--timeout','90',c['id']])
 ssh(c['node'],['docker','rename',c['id'],prefix+'-retired-'+run.name+'-'+c['suffix']])
 return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**c}
with (out/'actions.jsonl').open('w') as f:
 f.write(json.dumps(stop(plan[0]))+'\n');f.flush()
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(stop,plan[1:3]):f.write(json.dumps(result)+'\n');f.flush()
 for c in plan[3:]:f.write(json.dumps(stop(c))+'\n');f.flush()
with (out/'gpu-release.jsonl').open('w') as f:
 subprocess.run(['python3',str(root/'scripts/wait_nodes_idle.py'),prefill,decode,'--timeout','7200','--interval','30'],stdout=f,stderr=subprocess.STDOUT,check=True)
deadline=time.monotonic()+7200
with (out/'host-release.jsonl').open('w') as f:
 while True:
  state={}
  for node in (prefill,decode):
   raw=ssh(node,['cat','/proc/meminfo']);available=int(next(x for x in raw.splitlines() if x.startswith('MemAvailable:')).split()[1]);state[node]=available
  f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'available_kib':state})+'\n');f.flush()
  if all(v>=2*1024**3 for v in state.values()):break
  if time.monotonic()>deadline:raise RuntimeError('Host memory not released; do not restart')
  time.sleep(30)
(out/'RESOURCE_RELEASED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
print('RESOURCE_RELEASED',run.name,flush=True)
