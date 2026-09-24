#!/usr/bin/env python3
import json,os
from pathlib import Path
from allocation_state import snapshot
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN']);run.mkdir(parents=True,exist_ok=True)
r=snapshot(root,bind=True)
with (run/'allocation-checks.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
print(json.dumps({'passed':r['passed'],'errors':r['errors'],'jobs':{k:v['identity']['job'] for k,v in r['roles'].items()}}))
if not r['passed']:raise SystemExit(1)
