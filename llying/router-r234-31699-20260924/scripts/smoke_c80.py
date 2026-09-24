#!/usr/bin/env python3
"""C80 functional exercise only; results are not a throughput benchmark."""
import concurrent.futures,json,os,threading,time,urllib.request
from pathlib import Path
run=Path(os.environ['RUN']);barrier=threading.Barrier(80);lock=threading.Lock();active=0;peak=0

def request(i):
 global active,peak
 count=[400,800,1600,4000][i%4]
 body={'rid':f'r234-c80-smoke-{i}','model':'glm5.2-mxfp4','messages':[{'role':'user','content':'Read this test data and say OK.\n'+f'case {i}: temperature 20 pressure 100.\n'*count}],'max_tokens':32,'temperature':0,'stream':False}
 req=urllib.request.Request(f"http://{os.environ['PREFILL_IP']}:28000/v1/chat/completions",json.dumps(body).encode(),{'Content-Type':'application/json'})
 barrier.wait(timeout=30)
 with lock:active+=1;peak=max(peak,active)
 started=time.time()
 try:
  with urllib.request.urlopen(req,timeout=300) as response:result=json.load(response)
  assert result.get('choices') and result.get('usage',{}).get('completion_tokens',0)>0,result
  return {'rid':body['rid'],'passed':True,'seconds':time.time()-started,'response':result}
 except Exception as e:return {'rid':body['rid'],'passed':False,'error':str(e),'seconds':time.time()-started}
 finally:
  with lock:active-=1
with concurrent.futures.ThreadPoolExecutor(max_workers=80) as pool:rows=list(pool.map(request,range(80)))
report={'passed':all(x['passed'] for x in rows) and peak==80,'concurrency':80,'peak_client_active':peak,'rows':rows,'scope':'functional only, no performance gain inference'}
(run/'smoke-c80.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}))
if not report['passed']:raise SystemExit(1)
time.sleep(5)
