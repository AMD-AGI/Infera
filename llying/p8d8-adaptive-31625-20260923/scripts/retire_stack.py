#!/usr/bin/env python3
"""Retire only this experiment's audited, idle stack before a capacity restart."""
import argparse,datetime,json,os,re,shlex,subprocess,time,urllib.request
from pathlib import Path
from sample_engine_metrics import parse_metrics
p=argparse.ArgumentParser(description=__doc__);p.add_argument('completed_run',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
r=a.completed_run
if 'COMPLETE_REVIEW_PENDING' not in (r/'STATUS').read_text():raise SystemExit('Refuse: case not complete')
if not json.loads((r/'analysis/case-audit.json').read_text())['passed']:raise SystemExit('Refuse: case audit failed')
if (r/'INVALID').exists():raise SystemExit('Refuse: invalid case')
root=Path(os.environ['TRACE_RUNTIME']);prefix=os.environ['CONTAINER_PREFIX'];job=os.environ['ALLOCATION_JOB_ID'];opts=shlex.split(os.environ['SSH_OPTS'])
if prefix!=f'llying-adaptive-{job}':raise SystemExit('Refuse: unexpected prefix/job')
raw=subprocess.check_output(['scontrol','show','job',job],text=True,timeout=20);fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
nodes=set(subprocess.check_output(['scontrol','show','hostnames',fields['NodeList']],text=True,timeout=20).split())
if fields.get('JobState')!='RUNNING' or fields.get('PreemptTime') not in (None,'None') or not fields.get('UserId','').startswith('liyingli(') or nodes!={os.environ['PREFILL_NODE'],os.environ['DECODE_NODE']}:raise SystemExit('Refuse: allocation changed')
end=datetime.datetime.fromisoformat(fields['EndTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
review=datetime.datetime(2026,9,24,0,43,tzinfo=datetime.timezone.utc).timestamp()
if min(end-2400,review)-time.time()<10800:raise SystemExit('Refuse: under 3 hours before review/cleanup, do not start expensive restart')
def ssh(node,args):return subprocess.check_output(['ssh',*opts,node,shlex.join(args)],text=True,timeout=130)
expected='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
plan={'completed_case':str(r),'allocation':raw,'containers':[],'queues':{},'execute':a.execute}
queue_names={'sglang:num_queue_reqs','sglang:num_running_reqs','sglang:num_decode_prealloc_queue_reqs','sglang:num_decode_transfer_queue_reqs','sglang:num_prefill_bootstrap_queue_reqs','sglang:num_prefill_inflight_queue_reqs','sglang:num_prefill_prealloc_queue_reqs'}
ids=json.loads((r/'live-containers.json').read_text());router_id=json.loads((r/'snapshot/router-mode-validation.json').read_text())['container']['Id']
for role,port in [('prefill',29001),('decode',29002)]:
 node=os.environ[role.upper()+'_NODE'];ip=os.environ[role.upper()+'_IP']
 with urllib.request.urlopen(f'http://{ip}:{port}/metrics',timeout=30) as f:rows=parse_metrics(f.read().decode())[0]
 queue=[x for x in rows if x['metric'] in queue_names];ranks={x['labels'].get('dp_rank') for x in queue if x['metric']=='sglang:num_running_reqs'}
 if ranks!={str(i) for i in range(8)} or any(x['value']!=0 for x in queue):raise SystemExit(f'Refuse: {role} not verifiably idle')
 plan['queues'][role]=queue
 for name in ssh(node,['docker','ps','--format','{{.Names}}']).splitlines():
  if name.startswith(prefix+'-agentx-client-'):raise SystemExit('Refuse: benchmark client remains active')
for suffix,role in [('router','prefill'),('prefill-0','prefill'),('decode-0','decode'),('collector','prefill'),('etcd','prefill')]:
 node=os.environ[role.upper()+'_NODE'];name=prefix+'-'+suffix;c=json.loads(ssh(node,['docker','inspect',name]))[0]
 image_expected='sha256:13b135926ee29192305a1ab42861eca50d3b1b862869b8f4339743ad2fa7cc8e' if suffix=='etcd' else expected
 if c['Image']!=image_expected or c['Name']!='/'+name:raise SystemExit('Refuse: unexpected container identity/image')
 if suffix in ('prefill-0','decode-0') and c['Id']!=ids[role]['Id']:raise SystemExit('Refuse: engine replaced since capture')
 if suffix=='router' and c['Id']!=router_id:raise SystemExit('Refuse: router replaced since capture')
 plan['containers'].append({'node':node,'name':name,'id':c['Id'],'inspect':c,'retired_name':prefix+'-retired-'+r.name+'-'+suffix})
out=root/'events'/('retire-'+r.name);out.mkdir(parents=True,exist_ok=True)
(out/('execute-plan.json' if a.execute else 'dry-run-plan.json')).write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps({'execute':a.execute,'plan':str(out),'containers':[(x['node'],x['name']) for x in plan['containers']]}),flush=True)
if not a.execute:raise SystemExit(0)
for c in plan['containers']:
 started=time.time();result=ssh(c['node'],['docker','stop','--timeout','90',c['id']])
 ssh(c['node'],['docker','rename',c['id'],c['retired_name']])
 with (out/'actions.jsonl').open('a') as f:f.write(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'container':c['name'],'id':c['id'],'docker_stop_and_rename_s':time.time()-started,'stdout':result})+'\n')
print('Containers stopped; waiting for actual VRAM release, not just process exit.',flush=True)
with (out/'release-monitor.jsonl').open('w') as f:
 subprocess.run(['python3',str(root/'scripts/wait_nodes_idle.py'),os.environ['PREFILL_NODE'],os.environ['DECODE_NODE'],'--timeout','5400','--interval','30'],stdout=f,stderr=subprocess.STDOUT,check=True)
(out/'RESOURCE_RELEASED').write_text(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
print('RESOURCE_RELEASED',flush=True)
