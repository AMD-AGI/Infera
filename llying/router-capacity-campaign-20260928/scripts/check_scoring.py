"""Check recorded known P decisions against their complete candidate scores."""
import collections,json,os,re,shlex,subprocess
from pathlib import Path
run=Path(os.environ['RUN']);name=os.environ['CONTAINER_PREFIX']+'-router'
cmd=['ssh',*shlex.split(os.environ['SSH_OPTS']),os.environ['PREFILL_NODE'],shlex.join(['docker','logs',name])]
s=subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT)
s=re.sub(r'\x1b\[[0-9;]*m','',s);(run/'router-score-review.log').write_text(s)
groups=collections.defaultdict(list)
for line in s.splitlines():
 if 'routing experiment candidate' not in line or 'role=Prefill' not in line or 'dynamo_prefill=true' not in line:continue
 m=re.search(r'decision_id=(\d+)',line);cost=re.search(r'demand_cost=Some\(([\d.eE+-]+)\)',line)
 if m and cost and 'demand_known=true' in line:
  groups[m[1]].append((float(cost[1]),bool(re.search(r'(?<!_)selected=true',line))))
assert groups,'No known Dynamo P candidate decisions captured'
assert any('dynamo_prefill=true' in line and re.search(r'gpu_hits=[1-9][0-9]*',line) for line in s.splitlines()),'No GPU hit in the tiered directory after repeated requests'
for ident,rows in groups.items():
 chosen=[c for c,yes in rows if yes]
 assert len(chosen)==1 and chosen[0]<=min(c for c,_ in rows)+1e-9,(ident,rows)
assert not any('role=Decode' in line and 'dynamo_prefill=true' in line for line in s.splitlines())
result=dict(passed=True,known_decisions=len(groups),multi_candidate_decisions=sum(len(v)>1 for v in groups.values()),scope='Known P scores choose minimum; D remains outside the new mode. Pinned calls may have one candidate.')
assert result['multi_candidate_decisions']>0
(run/'score-review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
