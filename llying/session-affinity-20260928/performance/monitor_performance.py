"""One-shot observation; reports facts and never stops services."""
import datetime,json,os,re,subprocess,urllib.request
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928');run=root/'runs/session-affinity-31999-performance'
print(datetime.datetime.now(datetime.timezone.utc).isoformat())
env=dict(os.environ,SLURM_CONF='/home/liyingli/code/bench_agentx/Infera/llying/router-r234-31699-20260924/config/slurm-client-dccs.conf')
p=subprocess.run(['/opt/slurm/bin/squeue','-h','-j','31999','-o','%i %T %N %L'],env=env,text=True,capture_output=True);print('allocation:',p.stdout.strip() or p.stderr.strip() or 'not in queue')
for role,ip,port in [('prefill','10.235.192.140',29001),('decode','10.235.192.56',29002),('router','10.235.192.140',28000)]:
 try:
  with urllib.request.urlopen(f'http://{ip}:{port}/metrics',timeout=3) as f:s=f.read().decode()
  if role=='router':
   print(role,'; '.join(l for l in s.splitlines() if l.startswith('infera_router_session_')))
  else:
   result={}
   for key in ['num_running_reqs','num_queue_reqs','num_decode_transfer_queue_reqs','num_decode_prealloc_queue_reqs']:
    vals=[float(l.rsplit(' ',1)[1]) for l in s.splitlines() if re.match('sglang:'+key+r'(\{| )',l)]
    result[key]=sum(vals) if vals else None
   queues={}
   for line in s.splitlines():
    if line.startswith('sglang:num_queue_reqs{'):
     rank=re.search(r'dp_rank="([0-9]+)"',line)
     if rank:queues[rank.group(1)]=float(line.rsplit(' ',1)[1])
   if role=='prefill':result['queue_by_rank']=queues
   print(role,result)
 except Exception as e:print(role,type(e).__name__,str(e)[:100])
p=run/'logs/c80.log'
if p.exists():
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-14000));s=f.read().decode(errors='replace')
 s=re.sub(r'\x1b\[[0-9;]*m','',s).replace('\r','\n')
 lines=[l.strip() for l in s.splitlines() if l.strip()]
 important=[l for l in lines if any(k in l for k in ['Phase warmup progress', 'rps=', 'Phase profiling', 'Phase warmup (', 'ERROR', 'Traceback'])]
 print('client:', '\n'.join(important[-3:] or lines[-3:]))
if (run/'STATUS').exists():print('state:',(run/'STATUS').read_text().strip())
