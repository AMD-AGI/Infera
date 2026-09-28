"""Summarize complete client records without treating unfinished requests as errors."""
import argparse,collections,datetime,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
counts=collections.Counter();last={}
f=a.run/'c80/aiperf_artifacts/profile_export.jsonl'
if f.exists():
 for line in f.open():
  if not line.endswith('\n'):break
  r=json.loads(line);m=r.get('metadata',{});phase=m.get('benchmark_phase','unknown')
  counts[phase]+=1
  if m.get('was_cancelled'):counts[phase+'_cancelled']+=1
  last={k:m.get(k) for k in ['request_start_ns','request_end_ns','benchmark_phase','turn_index']}
state=(a.run/'STATUS').read_text().strip() if (a.run/'STATUS').exists() else 'STARTING'
x={'updated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'state':state,'complete_records_by_phase':dict(counts),'last_record':last}
(a.run/'live-summary.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
