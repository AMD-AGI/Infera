#!/usr/bin/env python3
"""Release this obsolete allocation only after its GPU/host memory is idle."""
import argparse,datetime,json,re,shlex,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--start',required=True);p.add_argument('--node',required=True);p.add_argument('--container-id',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
opts=['-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','UserKnownHostsFile=/tmp/agentx_known_hosts']
def record(x):
 x['utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 with (a.output/'events.jsonl').open('a') as f:f.write(json.dumps(x)+'\n')
def ssh(args,code=None):return subprocess.run(['ssh',*opts,a.node,shlex.join(args)],input=code,text=True,capture_output=True,timeout=40)
probe='''from pathlib import Path
import json
m={k:int(v.split()[0])*1024 for k,v in (l.split(':',1) for l in Path('/proc/meminfo').read_text().splitlines()) if k in ['MemAvailable','Shmem']}
g=[{'used':int((p/'mem_info_vram_used').read_text()),'total':int((p/'mem_info_vram_total').read_text()),'busy':int((p/'gpu_busy_percent').read_text())} for p in Path('/sys/class/drm').glob('card[0-9]*/device') if (p/'mem_info_vram_used').exists()]
print(json.dumps({'mem':m,'gpus':g}))'''
while True:
 try:
  q=subprocess.run(['scontrol','show','job',a.job],text=True,capture_output=True,timeout=15)
  if q.returncode:
   record({'event':'job_unavailable','detail':q.stderr});break
  f=dict(re.findall(r'(\w+)=([^\s]+)',q.stdout))
  if not f.get('UserId','').startswith('liyingli('):raise RuntimeError('Owner mismatch')
  if f.get('JobState') in ('COMPLETED','CANCELLED','FAILED','TIMEOUT'):
   record({'event':'job_already_ended','state':f['JobState'],'memory_release_not_verified':True});break
  if f.get('StartTime')!=a.start or f.get('JobState')=='PENDING':
   subprocess.run(['scancel',a.job],check=True);record({'event':'cancelled_unused_requeue','memory_release_on_previous_node_not_verified':True});break
  x=ssh(['python3','-'],probe)
  if x.returncode:
   record({'event':'probe_error','detail':x.stderr})
   if f.get('PreemptTime') not in (None,'None'):
    subprocess.run(['scancel',a.job],check=True);record({'event':'cancelled_after_access_revoked','memory_release_not_verified':True});break
  else:
   data=json.loads(x.stdout);g=data['gpus'];m=data['mem'];ready=len(g)==8 and all(z['used']<.02*z['total'] and z['busy']<=5 for z in g) and m['MemAvailable']>=2_000_000_000_000 and m['Shmem']<64*2**30
   record({'event':'memory_probe','ready':ready,**data})
   if ready:
    inspected=ssh(['docker','inspect',a.container_id])
    if inspected.returncode==0:
     c=json.loads(inspected.stdout)[0];assert c['Image']=='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
     if c['State']['Running']:
      stopped=ssh(['docker','stop','--timeout','5',a.container_id]);record({'event':'stop_after_gpu_release','returncode':stopped.returncode,'stderr':stopped.stderr})
      if stopped.returncode:time.sleep(30);continue
    subprocess.run(['scancel',a.job],check=True);record({'event':'released_allocation_after_verified_memory_release'});break
 except Exception as e:record({'event':'error','detail':str(e)})
 time.sleep(30)
