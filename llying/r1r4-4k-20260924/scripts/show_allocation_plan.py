#!/usr/bin/env python3
"""Read scheduling forecasts for the agent; never stop or start a service."""
import datetime,json,os,re,subprocess
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924')
os.environ['SLURM_CONF']=str(root/'config/slurm-client-dccs.conf')
def job(j):
 text=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job',str(j),'-o'],text=True,timeout=10)
 return dict(re.findall(r'(\w+)=([^\s]+)',text))
def nodes(spec):
 if not spec or spec in ['(null)','None']:return []
 return subprocess.check_output(['/opt/slurm/bin/scontrol','show','hostnames',spec],text=True,timeout=10).split()
own=job(31719);assigned=set(nodes(own.get('NodeList') or own.get('SchedNodeList')))
ids=subprocess.check_output(['/opt/slurm/bin/squeue','-p','Compute-DCPT','-t','PD','-h','-o','%i'],text=True,timeout=10).split()
policy=json.loads((root/'review/qos-preemption.json').read_text()) if (root/'review/qos-preemption.json').exists() else {}
preemptors=set(policy.get('batch_preemptors',[]))
plans=[]
for j in ids:
 if not j.isdigit() or j=='31719':continue
 f=job(j)
 if f.get('QOS')=='batch':continue
 scheduled=set(nodes(f.get('SchedNodeList') or f.get('ReqNodeList')))
 plans.append({'job':j,'qos':f.get('QOS'),'can_preempt_batch':f.get('QOS') in preemptors,'state':f.get('JobState'),'planned_start':f.get('StartTime'),'planned_end':f.get('EndTime'),'nodes':sorted(scheduled),'overlap_with_our_nodes':sorted(assigned&scheduled)})
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'own':{k:own.get(k) for k in ['JobId','JobState','StartTime','EndTime','PreemptTime','PreemptEligibleTime','Restarts','NodeList','SchedNodeList']},'non_batch_pending':plans,'note':'Forecasts may change; use for pre-launch planning, not as an automatic failure gate.'}
with (root/'events/scheduling-plans.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
print(json.dumps(report,indent=2))
