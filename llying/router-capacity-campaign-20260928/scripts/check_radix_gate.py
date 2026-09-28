"""Confirm the hit/miss gate exercised decode-local reuse with real acceptance."""
import json,os,shlex,subprocess
from pathlib import Path

r=Path(os.environ['RUN']);gate=json.loads((r/'radix-gate.json').read_text())
answer_gate=None
if not gate['pass']:
 answer_gate=json.loads((r/'radix-answer-gate.json').read_text())
 assert answer_gate['passed'],gate['checks']
info=json.loads((r/'launch/server-info/decode-0.json').read_text())
assert info['disaggregation_decode_enable_radix_cache'] and not info['disable_radix_cache']
assert info['speculative_algorithm']=='EAGLE'
c=json.loads((r/'snapshot/decode-container.json').read_text())
env=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
assert not env.get('SGLANG_SIMULATE_ACC_LEN'),'correctness gate requires real acceptance'
assert env.get('SGLANG_EXPERIMENTAL_DECODE_RADIX_SPEC')=='1'
directory=next(m['Source'] for m in c['Mounts'] if m['Destination']=='/aus-diag')
ids={trial[key]['rid'] for trial in gate['prefix_trials'] for key in ['r2_hit','r2_miss']}
code='''import json,sys
from pathlib import Path
wanted=set(json.loads(sys.argv[2]));out=[]
for p in Path(sys.argv[1]).glob('*.jsonl'):
 for line in p.open():
  x=json.loads(line)
  if x.get('event')=='decode_prefix' and x.get('rid') in wanted:out.append(x)
print(json.dumps(out))
'''
cmd=['ssh',*shlex.split(os.environ['SSH_OPTS']),os.environ['DECODE_NODE'],shlex.join(['python3','-c',code,directory,json.dumps(sorted(ids))])]
events=json.loads(subprocess.check_output(cmd,text=True));mapped={}
for x in events:
 assert x['rid'] not in mapped,'unexpected reallocation in serial gate'
 assert 0<=x['device_prefix_tokens']<=x['prefix_tokens']<=x['input_tokens']
 mapped[x['rid']]=x
assert set(mapped)==ids,(set(mapped),ids)
checks=[]
for t in gate['prefix_trials']:
 hit=mapped[t['r2_hit']['rid']];miss=mapped[t['r2_miss']['rid']]
 assert hit['prefix_tokens']>0,hit
 assert miss['prefix_tokens']==0,miss
 assert hit['dp_rank']==miss['dp_rank'],(hit,miss)
 checks.append({'rank':hit['dp_rank'],'hit_prefix':hit['prefix_tokens'],'miss_prefix':miss['prefix_tokens'],'greedy_output_equal':t['r2_hit_equal_miss']})
result={'passed':True,'real_acceptance':True,'strict_text_parity_passed':gate['pass'],'known_answer_gate_passed':answer_gate['passed'] if answer_gate else None,'trials':checks,'events':events}
(r/'radix-local-reuse-check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'trials':checks}))
