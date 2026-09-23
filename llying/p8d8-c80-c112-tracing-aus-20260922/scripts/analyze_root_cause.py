#!/usr/bin/env python3
"""Reproducible offline RCA: cohort demand, tails, cache groups and capacity checks.

Read-only on the run directory. Cohort demand is sum(duration)/3600, not a
boundary-clipped time average. Token-seconds use input length, exclude growth,
and are an occupancy proxy, not allocator telemetry.
"""
import argparse, collections, datetime, hashlib, json
from pathlib import Path
from analyze import jsonl, stats


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('run',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    rows=[r for r in jsonl(a.run/'analysis/joined-requests.jsonl') if r['phase']=='profiling']
    assert len({(r['concurrency'],r['rid']) for r in rows})==len(rows)
    blocked=collections.defaultdict(set); events=collections.defaultdict(list)
    for f in (a.run/'diagnostics/decode').glob('*.jsonl'):
        for r in jsonl(f):
            if r.get('event')=='admission' and r.get('reason')!='admitted':
                blocked[r['reason']].add(r['rid']);events[r['rid']].append(r)
    result={'method':__doc__,'points':{}}
    for c in (80,112):
        rs=[r for r in rows if r['concurrency']==c]; assert len(rs)=={80:9651,112:9321}[c]
        def metrics(group):
            n=len(group)
            return {'n':n, 'input_tokens':sum(r['prefill']['input_tokens'] for r in group),
                    'miss_tokens':sum(r['prefill']['input_tokens']-sum(r['prefill'][k] for k in ('cached_device','cached_host','cached_storage')) for r in group),
                    'host_tokens':sum(r['prefill']['cached_host'] for r in group),
                    'queue_ms':stats([r['prefill']['durations_ms']['queue_ms'] for r in group]),
                    'forward_ms':stats([r['prefill']['durations_ms']['forward_envelope_ms'] for r in group]),
                    'alloc_ms':stats([r['decode']['durations_ms']['alloc_wait_ms'] for r in group]),
                    'ttft_ms':stats([r['metrics']['time_to_first_token']['value'] for r in group])}
        groups={'all':rs,'host_hit':[r for r in rs if r['prefill']['cached_host']>0],
                'no_host_hit':[r for r in rs if r['prefill']['cached_host']==0],
                'kv_blocked':[r for r in rs if r['rid'] in blocked['kv_budget']],
                'not_kv_blocked':[r for r in rs if r['rid'] not in blocked['kv_budget']]}
        for lo,hi in [(0,1024),(1024,4096),(4096,8192),(8192,32768),(32768,10**9)]:
            groups[f'miss_{lo}_{hi}']=[r for r in rs if lo<=r['prefill']['input_tokens']-sum(r['prefill'][k] for k in ('cached_device','cached_host','cached_storage'))<hi]
        p={'groups':{k:metrics(v) for k,v in groups.items()},'stage_cohort_demand':{},'input_token_occupancy_proxy':{},'tail_work':{},'blocking':{}}
        for role in ('prefill','decode'):
            for stage in rs[0][role]['durations_ms']:
                ds=[r[role]['durations_ms'][stage] for r in rs]
                p['stage_cohort_demand'][role+'/'+stage]=sum(ds)/1000/3600
        for stage in ('transfer_wait_ms','queue_ms','generation_ms'):
            p['input_token_occupancy_proxy'][stage]=sum(r['decode']['input_tokens']*r['decode']['durations_ms'][stage]/1000 for r in rs)/3600
        for reason, ids in blocked.items():
            subset=[r for r in rs if r['rid'] in ids]
            es=[e for r in subset for e in events[r['rid']] if e['reason']==reason]
            p['blocking'][reason]={'requests':len(subset),'events':len(es),
                'free_requests':stats([e['free_requests'] for e in es]),
                'free_metadata':stats([e['free_metadata'] for e in es]),
                'required':stats([e['required'] for e in es]),'budget':stats([e['budget'] for e in es])}
        for frac in (.01,.05,.1):
            selected=sorted(rs,key=lambda r:r['prefill']['input_tokens']-sum(r['prefill'][k] for k in ('cached_device','cached_host','cached_storage')),reverse=True)[:max(1,round(len(rs)*frac))]
            p['tail_work'][str(frac)]=metrics(selected)
        for field in ('queue_ms','forward_envelope_ms'):
            vals=[r['prefill']['durations_ms'][field] for r in rs]
            p[field+'_fractions']={str(t):sum(x>t for x in vals)/len(vals) for t in (1,10,1000,10000,30000)}
        p['client_http_blocked_ms']=stats([r['metrics'].get('http_req_blocked',{}).get('value') for r in rs])
        result['points'][str(c)]=p
    summary=json.loads((a.run/'analysis/summary.json').read_text())
    windows={int(c):(p['phases']['profiling']['start_ns']/1e9,p['phases']['profiling']['start_ns']/1e9+3600) for c,p in summary['points'].items()}
    resources=collections.defaultdict(lambda:collections.defaultdict(list))
    for r in jsonl(a.run/'sampling/engine.jsonl'):
        if r.get('record_type')!='sample' or r.get('error') or r['endpoint'] not in ('prefill','decode'):continue
        t=datetime.datetime.fromisoformat(r['captured_at']).timestamp()
        c=next((c for c,(lo,hi) in windows.items() if lo<=t<hi),None)
        if c is None:continue
        byrank=collections.defaultdict(dict)
        for s in r['series']:
            if 'dp_rank' in s['labels']:byrank[str(s['labels']['dp_rank'])][s['metric']]=s['value']
        keys=('sglang:kv_available_tokens','sglang:kv_evictable_tokens','sglang:kv_used_tokens','sglang:max_total_num_tokens')
        if set(byrank)!=set(map(str,range(8))) or not all(all(k in v for k in keys) for v in byrank.values()):continue
        for rank,v in byrank.items():
            cap=v[keys[3]];free,evict,used=[v[k] for k in keys[:3]]
            assert abs(cap-free-evict-used)<=64, (cap,free,evict,used)
            group=resources[f'C{c}/{r["endpoint"]}']
            group['active_fraction'].append(used/cap);group['resident_fraction'].append((used+evict)/cap)
            group['reclaimable_tokens'].append(free+evict)
            group['queue_positive_and_reclaimable_ge_524288'].append(int(v.get('sglang:num_queue_reqs',0)>0 and free+evict>=524288))
            if v.get('sglang:num_queue_reqs',0)>0:group['reclaimable_when_queued'].append(free+evict)
    result['rank_sample_capacity']={k:{m:stats(v) for m,v in d.items()} for k,d in resources.items()}
    source_paths=['analysis/joined-requests.jsonl','sampling/engine.jsonl','analysis/summary.json']
    result['sources']={s:{'bytes':(a.run/s).stat().st_size,'sha256':digest(a.run/s)} for s in source_paths}
    (a.output/'root-cause-evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(a.output/'root-cause-evidence.json')

if __name__=='__main__':main()
