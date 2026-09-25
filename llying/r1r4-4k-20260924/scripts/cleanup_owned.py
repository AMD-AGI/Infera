#!/usr/bin/env python3
"""Start HiCache release before offline analysis; retain containers and allocation."""
import datetime,json,os,shlex,subprocess
from pathlib import Path
r=Path(os.environ['RUN']);root=Path(os.environ['TRACE_RUNTIME']);prefix=os.environ['CONTAINER_PREFIX'];out=r/'cleanup';out.mkdir(exist_ok=True)
live=json.load(open(r/'live-containers.json')) if (r/'live-containers.json').exists() else {};router=json.load(open(r/'snapshot/router-mode-validation.json'))['container'] if (r/'snapshot/router-mode-validation.json').exists() else {}
items=[('prefill-0','PREFILL',live.get('prefill',{}).get('Id')),('decode-0','DECODE',live.get('decode',{}).get('Id')),('router','PREFILL',router.get('Id')),('collector','PREFILL',None),('etcd','PREFILL',None)]
all_stopped=True
for suffix,role,expected in items:
 node=os.environ[role+'_NODE'];cmd=['ssh',*shlex.split(os.environ['SSH_OPTS']),node];name=prefix+'-'+suffix
 try:
  c=json.loads(subprocess.check_output(cmd+['docker','inspect',name],text=True,timeout=20))[0]
  if expected:assert c['Id']==expected
  assert c['Name']=='/'+name
  if suffix!='etcd':assert c['Image']=='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
  else:assert c['Config']['Image']=='quay.io/coreos/etcd:v3.5.14'
  event={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'name':name,'id':c['Id'],'event':'stop_requested'}
  with (out/'actions.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
  result=subprocess.run(cmd+['docker','stop','--timeout','30',c['Id']],capture_output=True,text=True,timeout=60)
  event.update(event='stop_returned',returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
  all_stopped = all_stopped and result.returncode == 0
 except Exception as e:
  all_stopped=False
  event={'name':name,'event':'stop_error','error':str(e)}
 with (out/'actions.jsonl').open('a') as f:f.write(json.dumps(event)+'\n')
# The independent guards remain until all intended services have stopped.

if all_stopped:
 # Keep the allocation guards while the client finishes offline exports.
 query=['ssh',*shlex.split(os.environ['SSH_OPTS']),os.environ['PREFILL_NODE'],shlex.join(['docker','ps','--filter','name='+prefix+'-agentx-client-','--format','{{.Names}}'])]
 client_names=subprocess.check_output(query,text=True,timeout=20).splitlines()
 if not client_names or (r/'c80-completed.txt').exists():
  (root/'events/cleanup-complete').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
