#!/usr/bin/env python3
"""Select one phase after the previous services have stopped."""
import json,sys
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924');phase=sys.argv[1]
assert phase in ['smoke','performance']
run=root/f'runs/r1r4-31719-{phase}'
(root/'config/active-phase.json').write_text(json.dumps({'prefix':f'llying-r1r4-31719-{phase}','run':str(run)},indent=2)+'\n')
(root/'active-run.txt').write_text(str(run)+'\n')
if (root/'events/cleanup-complete').exists():
 (root/'events/cleanup-complete').rename(root/'events/smoke-cleanup-complete')
for role in ['prefill','decode']:
 p=root/f'events/container-guard-{role}.jsonl'
 if p.exists():p.rename(root/f'events/container-guard-{role}-smoke.jsonl')
print(root/f'config/{phase}.sh')
