#!/usr/bin/env python3
"""Count request chunk envelopes; do not interpret them as exclusive GPU time."""
import argparse,collections,json
from analyze import jsonl,stats
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
rows={(r['concurrency'],r['rid']):r for r in jsonl(a.run/'analysis/joined-requests.jsonl') if r['phase']=='profiling'}
spans=collections.defaultdict(list)
for s in jsonl(a.run/'analysis/span-timelines.jsonl'):
 if s['phase']=='profiling' and s['role']=='prefill' and s['name']=='chunked_prefill':spans[(s['concurrency'],s['rid'])].append(s['duration_ms'])
result={}
for c in (80,112):
 r=[(k,v) for k,v in rows.items() if k[0]==c]
 groups={}
 for name,subset in [('all',r),('miss_ge32768',[(k,v) for k,v in r if v['prefill']['input_tokens']-sum(v['prefill'][x] for x in ('cached_device','cached_host','cached_storage'))>=32768])]:
  counts=[len(spans[k]) if bool(spans.get(k)) else 1 for k,v in subset]
  groups[name]={'requests':len(subset),'multi_chunk_requests':sum(bool(spans.get(k)) for k,v in subset),'observed_chunk_spans':sum(len(spans.get(k,[])) for k,v in subset),
   'passes_including_single_chunk_inference':sum(counts),'passes_per_request':stats(counts),
   'chunk_envelope_ms':stats([x for k,v in subset for x in spans.get(k,[])]),
   'chunk_forward_mismatch_gt1ms':sum(abs(sum(spans[k])-v['prefill']['durations_ms']['forward_envelope_ms'])>1 for k,v in subset if bool(spans.get(k))),
   'sum_chunk_vs_forward_max_abs_ms':max([abs(sum(spans[k])-v['prefill']['durations_ms']['forward_envelope_ms']) for k,v in subset if bool(spans.get(k))],default=0)}
 result[str(c)]=groups
(a.output/'chunk-evidence.json').write_text(json.dumps({'method':__doc__,'note':'Requests without chunked_prefill spans are inferred single-pass; multi-chunk envelopes include inter-completion gaps. Shared batches duplicate request spans.','points':result},indent=2)+'\n')
print(json.dumps(result,indent=2))
