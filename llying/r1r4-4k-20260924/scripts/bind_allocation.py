#!/usr/bin/env python3
"""Bind a granted allocation to this experiment; do not start services."""
import json, os, re, shlex, subprocess
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924')
os.environ['SLURM_CONF']=str(root/'config/slurm-client-dccs.conf')
job='31719'
raw=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job',job,'-o'],text=True)
f=dict(re.findall(r'(\w+)=([^\s]+)',raw))
if f.get('JobState')!='RUNNING':
 print(json.dumps({'job':job,'state':f.get('JobState'),'reason':f.get('Reason')}));raise SystemExit(2)
assert f.get('UserId','').endswith('(100078)'),f['UserId']
nodes=subprocess.check_output(['/opt/slurm/bin/scontrol','show','hostnames',f['NodeList']],text=True).split()
assert len(nodes)==2 and 'smci355-ccs-aus-n04-29' not in nodes,nodes
assigned={'job':job,'start':f['StartTime']}
for role,node in zip(['prefill','decode'],nodes):
 row=subprocess.check_output(['/opt/slurm/bin/scontrol','show','node',node,'-o'],text=True)
 nf=dict(re.findall(r'(\w+)=([^\s]+)',row))
 assigned[role]={'node':node,'ip':nf['NodeAddr']}
(root/'config/assigned-nodes.json').write_text(json.dumps(assigned,indent=2)+'\n')
(root/'config/nodes.sh').write_text('\n'.join(f'export {role.upper()}_{field}='+shlex.quote(assigned[role][key]) for role in ['prefill','decode'] for field,key in [('NODE','node'),('IP','ip')])+'\n')
(root/'config/topology.tsv').write_text('role\tnode\tdata_ip\n'+''.join(f"{role}\t{assigned[role]['node']}\t{assigned[role]['ip']}\n" for role in ['prefill','decode']))
print(json.dumps(assigned))
