"""Exercise every P rank with distinct long prefixes, then repeat each session."""
import json,os,subprocess,sys,uuid
from pathlib import Path
import radix_gate as probe
from analyze import jsonl

E=os.environ;RUN=Path(E['RUN']);workers=json.loads((RUN/'placement-resolved.json').read_text())
for w in workers:
    c=json.loads((RUN/f"snapshot/{w['instance']}-container.json").read_text())
    env=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
    if w['role']=='decode':assert not env.get('SGLANG_SIMULATE_ACC_LEN'),'probe requires real acceptance'
expected_p={(w['instance'],i) for w in workers if w['role']=='prefill' for i in range(w['dp'])}
expected_d={(w['instance'],i) for w in workers if w['role']=='decode' for i in range(w['dp'])}
tag=uuid.uuid4().hex;results=[];probe.GATE_LOG=RUN/'placement-probe-replies.jsonl'
for turn in range(2):
    for index in range(len(expected_p)):
        seed=index+1;identity=f'{tag}-{index}'
        doc='Registry '+identity+'\n'+'\n'.join(f'Record {j}: city {probe.CITIES[j%len(probe.CITIES)]}; code Z{(j*31337+seed*97)%9973:04d}.' for j in range(400))
        record=(index*17+311+turn*23)%400;answer=f'Z{(record*31337+seed*97)%9973:04d}'
        probe.SESSION_ID='placement-'+identity
        response=probe.chat('http://'+E['PREFILL_IP']+':28000',E['SERVED_MODEL'],[{'role':'system','content':doc},{'role':'user','content':f'What is the code in record {record}? Reply with only the code.'}],512)
        result={'session':identity,'turn':turn,'record':record,'expected':answer,**response};results.append(result)
        (RUN/'placement-answer-progress.json').write_text(json.dumps(results,indent=2)+'\n')
        assert response['content'].strip()==answer,result
subprocess.run([sys.executable,str(Path(__file__).parent/'capture_placement.py')],check=True)
by_role={role:{} for role in ['prefill','decode']};prefix={};ids={r['rid'] for r in results}
for role in by_role:
    for path in (RUN/'diagnostics'/role).glob('*.jsonl'):
        for event in jsonl(path):
            if event.get('rid') not in ids:continue
            if event['event']=='request_summary':by_role[role][event['rid']]=event
            if event['event']=='decode_prefix':prefix[event['rid']]=event
seen_p=set();seen_d=set();sessions={}
for r in results:
    rid=r['rid'];p=by_role['prefill'][rid];d=by_role['decode'][rid]
    assert p['room']==d['room']
    pt=(p['source_worker'],p['dp_rank']);dt=(d['source_worker'],d['dp_rank'])
    seen_p.add(pt);seen_d.add(dt)
    if r['session'] in sessions:
        previous=sessions[r['session']];assert pt==previous[0]
        if E['INFERA_SESSION_AFFINITY']=='both':assert dt==previous[1]
    sessions[r['session']]=(pt,dt)
    r['p_target']=pt;r['d_target']=dt
assert seen_p==expected_p,(seen_p,expected_p)
assert seen_d==expected_d,(seen_d,expected_d)
if E.get('DECODE_RADIX','0')=='1':assert any(x['prefix_tokens']>0 for x in prefix.values())
report={'passed':True,'real_acceptance':True,'correct_answers':len(results),'p_targets':sorted(seen_p),'d_targets':sorted(seen_d),'decode_prefix_events':list(prefix.values()),'requests':results,'limits':['Synthetic long-prefix retrieval and routing coverage; not a comprehensive accuracy benchmark.']}
(RUN/'placement-answer-probe.json').write_text(json.dumps(report,indent=2)+'\n')
(RUN/'STATUS').write_text('PLACEMENT_GATE_PASSED\n')
print(json.dumps({'passed':True,'answers':len(results),'p_targets':len(seen_p),'d_targets':len(seen_d)}))
