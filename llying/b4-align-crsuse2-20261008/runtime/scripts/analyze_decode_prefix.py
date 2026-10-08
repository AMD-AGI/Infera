"""Decode-local prefix accounting, independent of P cache statistics in D summaries."""
import argparse,collections,json
from pathlib import Path
from analyze import canonical,jsonl,stats

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
completed={x['rid']:x for x in jsonl(run/'analysis/joined-requests.jsonl') if x.get('phase')=='profiling' and x.get('decode')}
events=collections.defaultdict(list)
for path in (run/'diagnostics/decode').glob('*.jsonl'):
    for x in jsonl(path):
        rid=canonical(x.get('rid'))
        if x.get('event')=='decode_prefix' and rid in completed:
            assert 0<=x['device_prefix_tokens']<=x['prefix_tokens']<=x['input_tokens'],x
            events[rid].append(x)
rows=[]
for rid,es in events.items():
    es.sort(key=lambda x:x['wall_ns'])
    rows.append(dict(rid=rid,input_tokens=es[-1]['input_tokens'],last_prefix_tokens=es[-1]['prefix_tokens'],allocation_attempts=len(es),transfer_tokens_over_attempts=sum(x['input_tokens']-x['prefix_tokens'] for x in es),prefix_tokens_over_attempts=sum(x['prefix_tokens'] for x in es)))
result={'completed_requests':len(completed),'requests_with_local_prefix_event':len(rows),'event_coverage':len(rows)/len(completed) if completed else None,'requests_with_reuse':sum(r['last_prefix_tokens']>0 for r in rows),'requests_with_multiple_allocations':sum(r['allocation_attempts']>1 for r in rows),'input_tokens_last_allocation':sum(r['input_tokens'] for r in rows),'prefix_tokens_last_allocation':sum(r['last_prefix_tokens'] for r in rows),'logical_transfer_tokens_all_allocations':sum(r['transfer_tokens_over_attempts'] for r in rows),'prefix_tokens_distribution':stats([r['last_prefix_tokens'] for r in rows]),'limits':['Prefix tokens are selected by the D allocator and carried in the PD protocol; not measured RDMA bytes.','Only completed profiling requests are included. Retried allocations are counted separately.','D request-summary cached_device/host fields include forwarded P statistics and are not used here.']}
(run/'analysis/decode-local-prefix.json').write_text(json.dumps(result,indent=2)+'\n')
with (run/'analysis/decode-local-prefix-requests.jsonl').open('w') as f:
    for row in rows:f.write(json.dumps(row)+'\n')
print(json.dumps(result))
