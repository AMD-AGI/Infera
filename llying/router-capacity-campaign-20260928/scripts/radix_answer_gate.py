"""Verify known registry answers and matched-prefix execution, retaining text-parity failures."""
import json,os,shlex,subprocess,uuid
from pathlib import Path
import radix_gate as probe

E=os.environ;RUN=Path(E['RUN']);url='http://'+E['PREFILL_IP']+':28000'
probe.GATE_LOG=RUN/'radix-answer-replies.jsonl'
probe.SESSION_ID='radix-answer-'+uuid.uuid4().hex
original=json.loads((RUN/'radix-gate.json').read_text())
assert original['checks']['coherence'] and original['checks']['echo_exact'] and original['checks']['prefix_no_degenerate']
for trial in original['prefix_trials']:
    for name in ['r1','r1b']:assert trial[name]['content'].strip()=='Nairobi'
    for name in ['r2_hit','r2_miss']:assert trial[name]['content'].strip()=='Z2186'
rows=[];doc=probe.document(400)
for i in [17,311,0,399,128,256,63,199]:
    expected=f'Z{(i*31337)%9973:04d}'
    messages=[{'role':'system','content':doc},{'role':'user','content':f'What is the code in record {i}? Reply with only the code.'}]
    for repeat in range(2):
        result=probe.chat(url,E['SERVED_MODEL'],messages,512)
        row={'record':i,'repeat':repeat,'expected':expected,**result};rows.append(row)
        (RUN/'radix-answer-progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        assert result['content'].strip()==expected,row
c=json.loads((RUN/'snapshot/decode-container.json').read_text());directory=next(m['Source'] for m in c['Mounts'] if m['Destination']=='/aus-diag')
ids={q['rid'] for q in rows}
for t in original['prefix_trials']:
    ids.update(t[k]['rid'] for k in ['r2_hit','r2_miss'])
code='''import json,sys
from pathlib import Path
wanted=set(json.loads(sys.argv[2]));rows=[]
for p in Path(sys.argv[1]).glob('*.jsonl'):
 for line in p.open():
  x=json.loads(line)
  if x.get('event')=='decode_prefix' and x.get('rid') in wanted:rows.append(x)
print(json.dumps(rows))'''
cmd=['ssh',*shlex.split(E['SSH_OPTS']),E['DECODE_NODE'],shlex.join(['python3','-c',code,directory,json.dumps(sorted(ids))])]
events=json.loads(subprocess.check_output(cmd,text=True));byid={x['rid']:x for x in events}
assert len(events)==len(ids) and set(byid)==ids
for t in original['prefix_trials']:
    hit=byid[t['r2_hit']['rid']];miss=byid[t['r2_miss']['rid']]
    assert hit['prefix_tokens']>0 and miss['prefix_tokens']==0 and hit['dp_rank']==miss['dp_rank'],(hit,miss)
for q in rows[1:]:assert byid[q['rid']]['prefix_tokens']>0,q
result={'passed':True,'known_answers_correct':len(rows)+8,'additional_requests':len(rows),'prefix_reuse_events':events,'original_text_parity_passed':original['checks']['prefix_reuse_equal'],'limits':['The final answers match known registry values, including hit and miss requests.','Reasoning strings differ even across cold/cold repeats; bitwise output equivalence is not established.','This is a targeted functional gate, not a full long-context accuracy evaluation.']}
(RUN/'radix-answer-gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'passed':True,'known_answers_correct':result['known_answers_correct'],'prefix_events':len(events)}))
