#!/usr/bin/env python3
import json,os,time,urllib.request
from pathlib import Path
r=Path(os.environ['RUN']);url=f"http://{os.environ['PREFILL_IP']}:28000/v1/workers"
for attempt in range(60):
 with urllib.request.urlopen(url,timeout=10) as f:d=json.load(f)
 workers=d if isinstance(d,list) else d.get('workers',d.get('data',d.get('instances',[])))
 if isinstance(workers,list) and len(workers)==2:break
 time.sleep(1)
else:raise RuntimeError('Router did not discover both workers')
(r/'launch/workers.json').write_text(json.dumps(d,indent=2)+'\n')
(r/'launch/server-info').mkdir(exist_ok=True)
for role,port in [('prefill',29001),('decode',29002)]:
 with urllib.request.urlopen(f"http://{os.environ[role.upper()+'_IP']}:{port}/get_server_info",timeout=30) as f:data=json.load(f)
 (r/f'launch/server-info/{role}-0.json').write_text(json.dumps(data,indent=2)+'\n')
 print(role,{k:data.get(k) for k in ['chunked_prefill_size','random_seed','max_total_num_tokens','mem_fraction_static','hicache_ratio']})
