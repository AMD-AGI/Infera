#!/usr/bin/env python3
"""Join router P-completion observations to client completion on the same control host."""
import argparse,collections,datetime,json,re,statistics
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--partial',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args();r=a.run
ansi=re.compile(r'\x1b\[[0-9;]*m')
observed={};duplicates=0
for raw in (r/'server-logs/router.log').read_text(errors='replace').splitlines():
 line=ansi.sub('',raw)
 if 'prefill HTTP leg drained; guard lifecycle observed' not in line:continue
 rid=re.search(r'\brid=([^\s]+)',line);sep=re.search(r'\bseparated=(true|false)',line)
 if not rid or not sep:continue
 stamp_match=re.match(r'^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(\d+))?Z$',line.split()[0])
 if not stamp_match:continue
 seconds=int(datetime.datetime.fromisoformat(stamp_match[1]).replace(tzinfo=datetime.timezone.utc).timestamp())
 stamp=seconds*10**9+int((stamp_match[2] or '').ljust(9,'0')[:9])
 key=rid[1].strip('"').split('_')[-1]
 if key in observed:duplicates+=1
 observed[key]={'prefill_http_end_ns':int(stamp),'separated':sep[1]=='true'}
values=collections.defaultdict(list);coverage=collections.Counter();rows=[]
with (r/'c80/aiperf_artifacts/profile_export.jsonl').open() as f:
 for line in f:
  try:row=json.loads(line)
  except json.JSONDecodeError:
   if a.partial:continue
   raise
  meta=row.get('metadata',{});phase=meta.get('benchmark_phase','unknown');rid=str(meta.get('x_request_id','')).split('_')[-1]
  coverage[phase+'/client_records']+=1
  obs=observed.get(rid)
  if not obs:continue
  coverage[phase+'/matched']+=1
  end=meta.get('request_end_ns')
  if not end:continue
  delta=(end-obs['prefill_http_end_ns'])/1e6
  values[phase].append(delta)
  rows.append({'rid':rid,'phase':phase,'client_end_minus_router_prefill_end_ms':delta,**obs})
def stats(xs):
 if not xs:return {'n':0}
 xs=sorted(xs)
 def q(p):
  i=(len(xs)-1)*p;j=int(i);return xs[j]+(xs[min(j+1,len(xs)-1)]-xs[j])*(i-j)
 return {'n':len(xs),'mean_ms':statistics.mean(xs),'p50_ms':q(.5),'p90_ms':q(.9),'p99_ms':q(.99),'positive_fraction':sum(x>0 for x in xs)/len(xs)}
out=a.output or r/('progress' if a.partial else 'analysis');out.mkdir(parents=True,exist_ok=True)
(out/'guard-lifecycle-joined.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rows))
result={'partial':a.partial,'coverage':dict(coverage),'duplicate_router_observations':duplicates,'client_end_minus_prefill_http_end':{k:stats(v) for k,v in values.items()},'separated_observations':sum(x['separated'] for x in observed.values()),'total_observations':len(observed),'limitations':['Router log and client run on the same host, but log publication and HTTP completion add small observation delays.','Positive gap is a proxy for excess P request-lifetime accounting in control, not distinct-block load or a throughput benefit estimate.','Negative gaps may include client cancellation or P tail finishing after D; retain rather than clamp.']}
(out/'guard-lifecycle-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
