#!/usr/bin/env python3
"""Summarize chosen-rank routing accounting within the completed profiling window."""
import argparse,collections,datetime,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path);a=p.parse_args();r=a.run
assert (r/'c80-completed.txt').exists()
intervals=json.loads((r/'analysis/runtime-summary.json').read_text())['intervals'];start,end=next((x/1e9,y/1e9) for label,x,y in intervals if label=='C80/profiling')
groups=collections.defaultdict(list);ansi=re.compile(r'\x1b\[[0-9;]*m')
for raw in (r/'server-logs/router.log').open():
 line=ansi.sub('',raw)
 if 'active_blocks=' not in line or 'policy="kv-aware"' not in line:continue
 stamp=line.split()[0];ts=datetime.datetime.fromisoformat(stamp[:19]).replace(tzinfo=datetime.timezone.utc).timestamp()+float('0.'+stamp[20:-1])
 if not start<=ts<end:continue
 pairs=dict(re.findall(r'\b(role|picked|cache_hits|request_blocks|active_blocks|w_overlap)=([^\s]+)',line))
 if set(pairs)!={'role','picked','cache_hits','request_blocks','active_blocks','w_overlap'}:raise ValueError('Unparsed routing observation')
 row={k:float(pairs[k]) for k in ('cache_hits','request_blocks','active_blocks','w_overlap')}
 groups[pairs['role']].append(row);groups[pairs['role']+'/'+pairs['picked']].append(row)
def stat(xs):
 xs=sorted(xs);n=len(xs)
 return {'n':n,'mean':statistics.mean(xs),'p50':statistics.median(xs),'p90':xs[min(n-1,int(.9*n))],'max':xs[-1]}
summary={}
for key,rows in groups.items():
 summary[key]={'picks':len(rows),'metrics':{field:stat([x[field] for x in rows]) for field in rows[0]},'zero_hit_picks':sum(x['cache_hits']==0 for x in rows),'zero_block_picks':sum(x['request_blocks']==0 for x in rows),'sum_hits':sum(x['cache_hits'] for x in rows),'sum_request_blocks':sum(x['request_blocks'] for x in rows)}
result={'run':str(r),'profile_window_epoch_s':[start,end],'groups':summary,'limitations':['Pick window includes dispatched requests that may finish after the window; it is not the completed-request cohort.','Only chosen-rank active blocks are logged, before on_request_started; this is not the sum of all ranks or per-request held block-time.','Router hit estimates are not measured GPU/host cache hits.','Winner-only observations cannot reconstruct scores for alternative ranks or establish a routing counterfactual.']}
out=a.output or r/'analysis';out.mkdir(parents=True,exist_ok=True);(out/'router-picks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in summary.items() if '/' not in k}))
