#!/usr/bin/env python3
"""Associate independent P/D OTLP traces by client rid, preserving all events."""
import argparse
import collections
import json
from pathlib import Path
from analyze import canonical, jsonl, stats

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('run',type=Path)
args=parser.parse_args();root=args.run
clients={}
for c in (80,112):
    for row in jsonl(root/f'c{c}/aiperf_artifacts/profile_export.jsonl'):
        meta=row.get('metadata',{})
        clients[canonical(meta.get('x_request_id'))]=(c,meta.get('benchmark_phase','unknown'))

spans={}; duplicates=0
for row in jsonl(root/'traces/spans.jsonl'):
    if row.get('record_type')!='span':continue
    key=(row['trace_id'],row['span_id'])
    if key in spans:duplicates+=1
    spans[key]=row
memo={}
def context(key,seen=None):
    if key in memo:return memo[key]
    seen=set() if seen is None else seen
    if key in seen:return {}
    seen.add(key); row=spans[key]
    parent=(row['trace_id'],row.get('parent_span_id'))
    result=dict(context(parent,seen)) if parent in spans else {}
    attrs=row.get('attributes',{})
    for name in ('rid','dp_rank','tp_rank','bootstrap_room'):
        if name in attrs:result[name]=attrs[name]
    for role in ('prefill','decode'):
        if row.get('name','').lower().startswith(role+' '):result['role']=role
    memo[key]=result
    return result

out=root/'analysis';out.mkdir(exist_ok=True)
stage=collections.defaultdict(list);counts=collections.Counter();coverage=collections.defaultdict(set)
with (out/'span-timelines.jsonl').open('w') as stream:
    for key,row in spans.items():
        ctx=context(key);rid=canonical(ctx.get('rid'));cohort=clients.get(rid)
        if cohort is None:
            counts['unmatched_spans']+=1;continue
        c,phase=cohort;role=ctx.get('role','unknown');rank=ctx.get('dp_rank')
        duration=(row['end_time_ns']-row['start_time_ns'])/1e6
        if duration<0:
            counts['negative_duration']+=1;continue
        name=row['name']; counts['matched_spans']+=1
        if name.startswith(('prefill_','decode_','mooncake_')) or name=='tokenize':
            stage[f'C{c}/{phase}/{role}/{name}'].append(duration)
            coverage[f'C{c}/{phase}/{role}/{name}'].add(rid)
        stream.write(json.dumps(dict(concurrency=c,phase=phase,rid=rid,role=role,
            dp_rank=rank,name=name,trace_id=row['trace_id'],span_id=row['span_id'],
            start_ns=row['start_time_ns'],end_ns=row['end_time_ns'],duration_ms=duration,
            events=row.get('events',[]),attributes=row.get('attributes',{})),separators=(',',':'))+'\n')
result=dict(unique_spans=len(spans),duplicates=duplicates,counts=dict(counts),
    stages_ms={k:stats(v) for k,v in stage.items()},request_coverage={k:len(v) for k,v in coverage.items()},
    note='Unmatched spans include startup/smoke and incomplete roots; do not interpret all as exporter loss.')
(out/'span-summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(out/'span-summary.json')
