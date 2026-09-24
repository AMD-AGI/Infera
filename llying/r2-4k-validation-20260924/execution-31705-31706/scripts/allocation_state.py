"""Read-only checks for the two independently held allocations."""
import datetime,json,os,re,subprocess
from pathlib import Path

def snapshot(root, bind=False):
 root=Path(root);binding=root/'config/allocation-identities.json'
 expected=json.loads(binding.read_text()) if binding.exists() else {}
 specs={'prefill':('31705','smci355-ccs-aus-n10-29'),'decode':('31706','smci355-ccs-aus-n01-21')}
 roles={};errors=[]
 for role,(job,node) in specs.items():
  raw=subprocess.check_output(['/opt/slurm/bin/scontrol','show','job',job,'-o'],text=True,timeout=10)
  fields=dict(re.findall(r'(\w+)=([^\s]+)',raw))
  nodes=subprocess.check_output(['/opt/slurm/bin/scontrol','show','hostnames',fields['NodeList']],text=True,timeout=10).split()
  if fields.get('JobState')!='RUNNING' or fields.get('PreemptTime') not in (None,'None'):errors.append(role+': allocation not usable')
  if not fields.get('UserId','').endswith('(100078)') or node not in nodes:errors.append(role+': owner/node mismatch')
  identity={'job':job,'node':node,'start':fields.get('StartTime')}
  if role in expected and expected[role]!=identity:errors.append(role+': allocation restarted or changed')
  end=fields.get('EndTime')
  if end and end not in ('Unknown','None'):
   remaining=datetime.datetime.fromisoformat(end).replace(tzinfo=datetime.timezone.utc).timestamp()-datetime.datetime.now(datetime.timezone.utc).timestamp()
   if remaining<600:errors.append(role+': allocation expires within 10 minutes')
  roles[role]={'identity':identity,'fields':fields}
 report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'passed':not errors,'errors':errors,'roles':roles}
 if bind and not expected and not errors:binding.write_text(json.dumps({k:v['identity'] for k,v in roles.items()},indent=2)+'\n')
 return report
