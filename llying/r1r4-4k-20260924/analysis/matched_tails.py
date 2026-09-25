"""Summarize cache-loss tails and same-cache request subsets against G0."""
import argparse
import importlib.util
import json
import statistics
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('candidate',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
module=Path(__file__).with_name('compare_matched_requests.py')
spec=importlib.util.spec_from_file_location('matched',module)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
left,_=m.load(Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/g0-guard-completion'))
right,_=m.load(a.candidate)
pairs=[]
for key in left.keys() & right.keys():
    x,y=left[key],right[key]
    if not all(r.get('prefill') and r.get('decode') for r in (x,y)):continue
    u,v=m.values(x),m.values(y)
    if abs(u['input_tokens']-v['input_tokens'])>min(8,.001*max(u['input_tokens'],v['input_tokens'])):continue
    if u['output_tokens']!=v['output_tokens']:continue
    pairs.append((u,v))
expected=json.loads((a.candidate/'analysis/matched-g0/matched-requests.json').read_text())['matched_pairs']
assert len(pairs)==expected

def summarize(rows):
    result={'n':len(rows)}
    for k in ['miss_tokens','ttft_ms','prefill_queue_ms','prefill_forward_envelope_ms','host_tokens']:
        result[k]={'G0_mean':statistics.fmean(u[k] for u,v in rows),'R1R4_mean':statistics.fmean(v[k] for u,v in rows)} if rows else None
    return result

deltas=[v['miss_tokens']-u['miss_tokens'] for u,v in pairs]
positive=sum(max(d,0) for d in deltas)
report={'matched_pairs':len(pairs),'net_extra_miss_tokens':sum(deltas),'gross_extra_miss_tokens':positive,
        'gross_saved_miss_tokens':sum(max(-d,0) for d in deltas),'median_miss_delta':statistics.median(deltas),
        'cache_loss_tails':{str(t):{'requests':sum(d>t for d in deltas),'extra_tokens':sum(d for d in deltas if d>t),'share_of_gross_extra':sum(d for d in deltas if d>t)/positive} for t in [64,4096,32768,100000]},
        'same_miss_within64':summarize([(u,v) for u,v in pairs if abs(v['miss_tokens']-u['miss_tokens'])<=64]),
        'same_device_and_host_within64':summarize([(u,v) for u,v in pairs if abs(v['device_tokens']-u['device_tokens'])<=64 and abs(v['host_tokens']-u['host_tokens'])<=64]),
        'scope':'Selected intersection of completed source turns. Same lengths do not prove byte-identical prompts. Queue/forward envelopes include scheduling effects; not an isolated GPU-time comparison.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
