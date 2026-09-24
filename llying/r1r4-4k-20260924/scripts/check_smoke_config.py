#!/usr/bin/env python3
import json, os
from pathlib import Path
r=Path(os.environ['RUN'])
checks={}
for role in ['prefill','decode']:
 d=json.loads((r/f'launch/server-info/{role}-0.json').read_text())
 checks[role]={'chunk':d.get('chunked_prefill_size'),'tp':d.get('tp_size'),'dp':d.get('dp_size'),'hicache':d.get('enable_hierarchical_cache')}
 assert checks[role]['chunk']==4096 and checks[role]['tp']==8 and checks[role]['dp']==8, checks
 assert not checks[role]['hicache'], checks
(r/'smoke-config.json').write_text(json.dumps(checks,indent=2)+'\n')
print(checks)
