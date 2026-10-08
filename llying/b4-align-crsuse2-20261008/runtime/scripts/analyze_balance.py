"""Use complete per-scrape rank vectors; active KV already excludes evictable cache."""
import argparse,collections,datetime,json,statistics
from pathlib import Path
from analyze import jsonl,stats

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
phase=json.loads((run/'analysis/summary.json').read_text())['points']['80']['phases']['profiling']
start=phase['start_ns']/1e9;end=start+3600
placement=run/'placement-resolved.json'
if placement.exists():layouts={w['instance']:(w['role'],w['dp']) for w in json.loads(placement.read_text())}
else:
    env=dict(l.split('=',1) for l in (run/'c80/runtime.env').read_text().splitlines() if '=' in l and not l.startswith('#'))
    layouts={role:(role,int(env[role.upper()+'_DP_SIZE'])) for role in ['prefill','decode']}
frames=collections.defaultdict(dict)
for row in jsonl(run/'sampling/engine.jsonl'):
    endpoint=row.get('endpoint')
    if endpoint not in layouts or row.get('error'):continue
    t=datetime.datetime.fromisoformat(row['captured_at']).timestamp()
    if not start<=t<end:continue
    metrics=collections.defaultdict(dict)
    for s in row.get('series',[]):
        rank=s.get('labels',{}).get('dp_rank')
        if rank is not None:metrics[s['metric']][int(rank)]=s['value']
    frames[row['captured_at']][endpoint]=metrics
result={}
keys=['kv_used_tokens','kv_evictable_tokens','max_total_num_tokens','num_running_reqs','num_queue_reqs','num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs']
for role in ['prefill','decode']:
    expected={name:dp for name,(kind,dp) in layouts.items() if kind==role};samples=[]
    for frame in frames.values():
        vectors={k:[] for k in keys};complete=True
        for name,dp in expected.items():
            m=frame.get(name,{})
            for k in keys:
                values=m.get('sglang:'+k,{})
                if set(values)!=set(range(dp)):complete=False;break
                vectors[k].extend(values[i] for i in range(dp))
            if not complete:break
        if not complete:continue
        cap=vectors['max_total_num_tokens']
        if not all(v>0 for v in cap):continue
        used=vectors['kv_used_tokens'];evict=vectors['kv_evictable_tokens'];batch=vectors['num_running_reqs'];q=vectors['num_queue_reqs']
        fractions=[v/c for v,c in zip(used,cap)]
        cv=lambda xs:statistics.pstdev(xs)/statistics.fmean(xs) if sum(xs)>0 else 0
        samples.append({'active_fraction':sum(used)/sum(cap),'resident_fraction':sum(v+e for v,e in zip(used,evict))/sum(cap),'max_active_fraction':max(fractions),'kv_cv':cv(used),'running_total':sum(batch),'batch_cv':cv(batch),'queue_total':sum(q),'queue_max':max(q),'hot_with_spare':int(max(fractions)>.9 and min(fractions)<.5),'queue4_with_other_empty':int(max(q)>=4 and min(q)==0),'prealloc_total':sum(vectors['num_decode_prealloc_queue_reqs']),'transfer_total':sum(vectors['num_decode_transfer_queue_reqs'])})
    result[role]={'complete_samples':len(samples),'rank_count':sum(expected.values()),'metrics':{k:stats([r[k] for r in samples]) for k in samples[0]} if samples else {}}
result['limits']=['Samples are about 2s apart and endpoint scrapes are sequential within a timestamp.','Incomplete rank vectors are excluded, not zero-filled.','kv_used_tokens is active/non-evictable; resident=used+evictable.','An empty local P queue does not imply an idle independent GPU.']
(run/'analysis/balance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
