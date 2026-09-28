"""Locate B1 cache losses in observed session starts and continuation requests."""
import collections,json,statistics
from pathlib import Path
from compare_matched_requests import KEYS,values

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
BASE=ROOT.parent/'session-affinity-31999-20260928/runs/session-affinity-31999-performance'
RUN=ROOT/'runs/campaign-b1-dynamo-p'


def load(path):
    all_rows=[json.loads(l) for l in (path/'analysis/joined-requests.jsonl').open()]
    all_rows.sort(key=lambda r:r['metadata']['request_start_ns'])
    seen=set();rows={};duplicate=set()
    for r in all_rows:
        m=r['metadata'];session=m['x_correlation_id'];first=session not in seen;seen.add(session)
        if r['phase']!='profiling' or not r.get('prefill') or not r.get('decode'):continue
        key=tuple(m.get(k) for k in KEYS)
        if key in rows:duplicate.add(key)
        rows[key]=(r,first)
    for key in duplicate:rows.pop(key,None)
    return rows


left=load(BASE);right=load(RUN);pairs=[];groups=collections.defaultdict(list)
for key in left.keys()&right.keys():
    a,first_a=left[key];b,first_b=right[key];x=values(a);y=values(b)
    if abs(x['input_tokens']-y['input_tokens'])>8 or abs(x['input_tokens']-y['input_tokens'])>.001*max(x['input_tokens'],y['input_tokens']) or x['output_tokens']!=y['output_tokens']:continue
    r={'identity':list(key),'reference_rid':a['rid'],'candidate_rid':b['rid'],'reference_rank':a['prefill']['dp_rank'],'candidate_rank':b['prefill']['dp_rank'],'reference_first_observed':first_a,'candidate_first_observed':first_b,'reference':x,'candidate':y,'delta_miss':y['miss_tokens']-x['miss_tokens'],'delta_queue_ms':y['prefill_queue_ms']-x['prefill_queue_ms']}
    group='first_in_both' if first_a and first_b else 'continuation_in_both' if not first_a and not first_b else 'different_observation_boundary'
    groups[group].append(r);pairs.append(r)
summary={}
for name,rs in groups.items():
    summary[name]={'pairs':len(rs),'delta_miss_sum':sum(r['delta_miss'] for r in rs),'delta_queue_ms_sum':sum(r['delta_queue_ms'] for r in rs),'reference_miss_mean':statistics.mean(r['reference']['miss_tokens'] for r in rs),'candidate_miss_mean':statistics.mean(r['candidate']['miss_tokens'] for r in rs)}
positive=sum(max(0,r['delta_miss']) for r in pairs)
tails=[r for r in pairs if r['delta_miss']>32768]
report={'matched_pairs':len(pairs),'groups':summary,'extra_miss_gt32k_pairs':len(tails),'extra_miss_gt32k_fraction_of_positive_delta':sum(r['delta_miss'] for r in tails)/positive if positive else None,'net_extra_miss':sum(r['delta_miss'] for r in pairs),'top_extra_miss_requests':sorted(pairs,key=lambda r:-r['delta_miss'])[:20],'limits':['First means first observed x_correlation_id request across warmup and profiling, not a logged binding reason.','Initial placement also affects later cache retention; continuation losses are not proof of migration.','Matched completed requests are a selected intersection.']}
expected=json.loads((RUN/'analysis/matched-baseline/matched-requests.json').read_text())
assert report['matched_pairs']==expected['matched_pairs']
assert abs(report['net_extra_miss']/len(pairs)-expected['metrics']['miss_tokens']['paired_delta_mean'])<1e-6
(RUN/'analysis/selection-cache-loss.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='top_extra_miss_requests'},indent=2))
