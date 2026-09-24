#!/usr/bin/env python3
"""Bounded correctness probes after stopping the long client; no throughput test."""
import concurrent.futures,json,os,threading,time,urllib.request
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924');out=root/'observation';out.mkdir(exist_ok=True)
url='http://10.235.192.61:28000/v1/chat/completions'
def run(rid,content,barrier=None):
 body={'rid':rid,'model':'glm5.2-mxfp4','messages':[{'role':'user','content':content}],'max_tokens':32,'temperature':0,'stream':False}
 req=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
 if barrier:barrier.wait()
 started=time.time_ns()
 with urllib.request.urlopen(req,timeout=180) as f:response=json.load(f)
 return {'rid':rid,'start_ns':started,'end_ns':time.time_ns(),'usage':response.get('usage',{}),'has_choices':bool(response.get('choices')),'body':body}
rows=[run('obs-empty-before','Reply OK for this accounting check.')]
barrier=threading.Barrier(4)
counts=[10,100,300,600]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 rows+=list(pool.map(lambda i:run(f'obs-demand-{i}',f'Accounting check {i}.\n'+f'Row {i}: temperature 20 pressure 100.\n'*counts[i],barrier),range(4)))
time.sleep(2)
rows.append(run('obs-cache-repeat',rows[4]['body']['messages'][0]['content']))
time.sleep(2)
rows.append(run('obs-empty-after','Reply OK for this final accounting check.'))
assert all(r['has_choices'] for r in rows)
(out/'probes.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps([{'rid':r['rid'],'usage':r['usage'],'seconds':(r['end_ns']-r['start_ns'])/1e9} for r in rows]),flush=True)
