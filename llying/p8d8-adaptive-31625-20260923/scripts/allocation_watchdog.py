#!/usr/bin/env python3
"""Node-local cleanup of this job's containers on preemption or lease expiry."""
import argparse,datetime,json,re,signal,subprocess,time,concurrent.futures
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--prefix',required=True);p.add_argument('--role',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--allocation-start');p.add_argument('--node');p.add_argument('--active-run-file',type=Path);a=p.parse_args()
assert re.fullmatch(r'llying-adaptive-\d+',a.prefix) and a.prefix.endswith(a.job)
a.root.mkdir(parents=True,exist_ok=True);events=a.root/f'allocation-watchdog-{a.role}.jsonl';requested=None;end=None
expected='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
def record(x):
 x['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with events.open('a') as f:f.write(json.dumps(x)+'\n')
def stop(signum,*_):
 global requested;requested=f'signal_{signum}'
signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
def cleanup(reason):
 record({'event':'cleanup_requested','reason':reason})
 if a.active_run_file and a.active_run_file.exists():
  active=Path(a.active_run_file.read_text().strip())
  phase=(active/'STATUS').read_text() if (active/'STATUS').exists() else ''
  if active.is_dir() and active.parent.name=='runs' and 'COMPLETE_REVIEW_PENDING' not in phase:
   (active/'INVALID').write_text('Allocation cleanup: '+reason+'\n')
 names=subprocess.check_output(['docker','ps','--format','{{.Names}}'],text=True).splitlines()
 names=[n for n in names if n.startswith(a.prefix+'-')]
 # Stop clients first; retain container metadata and logs (no rm).
 names.sort(key=lambda n:(0 if '-agentx-client-' in n else 1 if n.endswith('-router') else 3 if n.endswith('-collector') else 2,n))
 selected=[]
 for name in names:
  c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
  image= 'sha256:13b135926ee29192305a1ab42861eca50d3b1b862869b8f4339743ad2fa7cc8e' if name==a.prefix+'-etcd' else expected
  if c['Image']!=image:record({'event':'unexpected_image_refuse_cleanup','container':name});continue
  selected.append((name,c['Id']))
  record({'event':'container_stop_requested','container':name,'id':c['Id'],'timeout_s':5})
 # Preemption can revoke extern processes within seconds: submit all owned stops promptly.
 def stop_container(item):
  name,cid=item
  try:
   x=subprocess.run(['docker','stop','--timeout','5',cid],capture_output=True,text=True,timeout=20)
   return {'event':'container_stopped','container':name,'id':cid,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
  except Exception as e:return {'event':'container_stop_error','container':name,'error':str(e)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  futures=[pool.submit(stop_container,item) for item in selected]
  for future in concurrent.futures.as_completed(futures):record(future.result())

while True:
 if requested:cleanup(requested);break
 try:
  text=subprocess.check_output(['scontrol','show','job',a.job],text=True,timeout=5)
  fields=dict(re.findall(r'(\w+)=([^\s]+)',text))
  if fields.get('EndTime') not in (None,'Unknown'):
   end=datetime.datetime.fromisoformat(fields['EndTime']).replace(tzinfo=datetime.timezone.utc).timestamp()
  reason=None
  if a.allocation_start and fields.get('StartTime')!=a.allocation_start:reason='allocation_restarted'
  elif a.node and a.node not in subprocess.check_output(['scontrol','show','hostnames',fields['NodeList']],text=True,timeout=5).split():reason='node_no_longer_allocated'
  elif fields.get('PreemptTime') not in (None,'None'):reason='preemption_announced'
  elif fields.get('JobState') not in ('RUNNING','COMPLETING'):reason='allocation_not_running'
  elif end and time.time()>=end-2400:reason='lease_cleanup_margin_40_minutes'
  if reason:cleanup(reason);break
 except Exception as e:
  record({'event':'controller_query_error','error':str(e)})
  if end and time.time()>=end-2400:cleanup('cached_lease_cleanup_margin');break
 for _ in range(5):
  if requested:break
  time.sleep(1)
