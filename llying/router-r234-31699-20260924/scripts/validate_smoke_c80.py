#!/usr/bin/env python3
import json,os
from pathlib import Path
r=Path(os.environ['RUN']);expected={f'r234-c80-smoke-{i}' for i in range(80)};pairs={}
for p in (r/'diagnostics').rglob('*.jsonl'):
 for line in open(p):
  d=json.loads(line)
  if d.get('event')=='request_summary' and d.get('rid') in expected:pairs.setdefault(d['rid'],{})[d['role']]=d
errors=[]
for rid in expected:
 pair=pairs.get(rid,{})
 if set(pair)!={'prefill','decode'}:errors.append('Missing pair '+rid);continue
 if pair['prefill']['room']!=pair['decode']['room']:errors.append('Room mismatch '+rid)
 if pair['prefill']['input_tokens']!=pair['decode']['input_tokens']:errors.append('Input mismatch '+rid)
report={'passed':not errors,'errors':errors,'paired_requests':len(pairs),'ranks':{role:sorted({x[role]['dp_rank'] for x in pairs.values() if role in x}) for role in ['prefill','decode']}}
(r/'smoke-c80-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if errors:raise SystemExit(1)
