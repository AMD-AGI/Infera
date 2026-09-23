#!/usr/bin/env python3
import json,os,time,urllib.request
from pathlib import Path
r=Path(os.environ['RUN']);body={'rid':'guard-stream-smoke','model':'glm5.2-mxfp4','messages':[{'role':'user','content':'State in one short sentence that this is a test.'}],'max_tokens':16,'temperature':0,'stream':True}
req=urllib.request.Request(f"http://{os.environ['PREFILL_IP']}:28000/v1/chat/completions",json.dumps(body).encode(),{'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as response:data=response.read().decode()
chunks=[]
for line in data.splitlines():
 if line.startswith('data:') and line[5:].strip()!='[DONE]':
  x=json.loads(line[5:]);chunks.append(x)
assert any(c.get('choices') and any(v.get('delta',{}).get('content') or v.get('delta',{}).get('reasoning_content') for v in c['choices']) for c in chunks),data
(r/'stream-smoke.json').write_text(json.dumps({'passed':True,'chunks':chunks},indent=2)+'\n')
time.sleep(5)
