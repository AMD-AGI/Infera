#!/usr/bin/env python3
import collections,json,os,re,subprocess
from pathlib import Path
r=Path(os.environ['RUN']);name=os.environ['CONTAINER_PREFIX']+'-router'
container=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
env=dict(x.split('=',1) for x in container['Config']['Env'] if '=' in x)
keys=['INFERA_R2_DECODE_DEMAND','INFERA_R3_CACHE_TIERS','INFERA_R3_HOST_WEIGHT','INFERA_R4_PREFILL_WORK']
errors=[f'Wrong router env: {k}' for k in keys if env.get(k)!=os.environ[k]]
text=subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT,text=True)
text=re.sub(r'\x1b\[[0-9;]*m','',text)
rows=[v for v in text.splitlines() if 'routing experiment candidate' in v]
expected=collections.Counter(x['response']['usage']['prompt_tokens'] for x in json.loads((r/'smoke.json').read_text()))
counts={}
for role,var in [('Decode','INFERA_R2_DECODE_DEMAND'),('Prefill','INFERA_R4_PREFILL_WORK')]:
 if os.environ[var]=='off':continue
 selected=[v for v in rows if f'role={role}' in v and re.search(r'(?<!\w)selected=true\b',v)]
 values=collections.Counter(int(m[1]) for v in selected if (m:=re.search(r'input_tokens=Some\((\d+)\)',v)))
 counts[role]={'selected_known_lengths':dict(values),'expected_smoke_lengths':dict(expected),'rows':len(selected)}
 if any(values[k]<n for k,n in expected.items()):errors.append(role+' Router input lengths do not cover engine smoke prompt lengths')
 if not any('demand_known=true' in v for v in selected):errors.append(role+' has no valid demand decisions')
host_stores=max([int(x) for x in re.findall(r'host_stores: (\d+)',text)] or [0])
host_hits=max([int(x) for x in re.findall(r'host_hits=(\d+)',text)] or [0])
unknown=max([int(x) for x in re.findall(r'unknown_events: (\d+)',text)] or [0])
if os.environ['INFERA_R3_CACHE_TIERS']!='off':
 if not host_stores:errors.append('No live host-tier store events observed')
 if unknown:errors.append('Unrecognized cache-tier events observed')
report={'passed':not errors,'errors':errors,'environment':{k:env.get(k) for k in keys},'counts':counts,'max_host_stores':host_stores,'max_host_hits':host_hits,'max_unknown_events':unknown,'host_hits_note':'Exclusive host hits may require a larger working set; measured again after benchmark.'}
(r/'experiment-smoke-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
if errors:raise SystemExit(1)
