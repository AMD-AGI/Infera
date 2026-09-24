#!/usr/bin/env python3
"""Capture raw one-token responses without changing formal benchmark settings."""
import concurrent.futures, json, os, time, urllib.request
from pathlib import Path
out=Path(os.environ['RUN'])/'empty-response-probes';out.mkdir(exist_ok=True)
url=f"http://{os.environ['PREFILL_IP']}:28000/v1/chat/completions"
prompts=['Say hello.','Calculate 2+2.','Write a Python function returning 1.','Explain why the sky is blue.']

def probe(spec):
 i,prompt,tokens,skip,stream=spec
 body={'model':os.environ['SERVED_MODEL'],'rid':f'r1r4-empty-probe-{i}','messages':[{'role':'user','content':prompt}],'max_tokens':tokens,'temperature':0,'stream':stream,'skip_special_tokens':skip}
 if stream:body['stream_options']={'include_usage':True}
 start=time.time()
 try:
  request=urllib.request.Request(url,json.dumps(body).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(request,timeout=120) as response:status=response.status;raw=response.read().decode()
  parts=[]
  if stream:
   for line in raw.splitlines():
    if line.startswith('data: ') and line[6:]!='[DONE]':parts.append(json.loads(line[6:]))
  else:parts=[json.loads(raw)]
  visible=[];usage=[]
  for part in parts:
   if part.get('usage'):usage.append(part['usage'])
   for choice in part.get('choices',[]):
    data=choice.get('delta') or choice.get('message') or {}
    visible.extend(str(data[k]) for k in ['content','reasoning_content','reasoning','tool_calls'] if data.get(k))
  result={'spec':body,'status':status,'raw':raw,'visible_text':''.join(visible),'usage':usage,'elapsed_s':time.time()-start}
 except Exception as e:result={'spec':body,'error':repr(e),'elapsed_s':time.time()-start}
 (out/f'{i:03}.json').write_text(json.dumps(result,indent=2)+'\n')
 return {'i':i,'max_tokens':tokens,'skip_special_tokens':skip,'stream':stream,'empty':not result.get('visible_text'),'error':result.get('error'),'usage':result.get('usage')}
specs=[]
for prompt in prompts:
 for tokens in [1,16]:
  for skip in [True,False]:
   for stream in [True,False]:specs.append((len(specs),prompt,tokens,skip,stream))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(probe,specs))
(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'requests':len(rows),'empty':sum(x['empty'] for x in rows),'errors':sum(bool(x['error']) for x in rows)}))

# Force a known one-token reasoning delimiter, then compare the native text API
# with the OpenAI chat reasoning parser. These requests are smoke-only.
forced=[]
for native in [False,True]:
 endpoint=f"http://{os.environ['PREFILL_IP']}:28000"+('/generate' if native else '/v1/chat/completions')
 body={'model':os.environ['SERVED_MODEL'],'rid':f'r1r4-forced-think-{native}','stream':False}
 if native:
  body.update(text='Say hello.',sampling_params={'max_new_tokens':1,'temperature':0,'skip_special_tokens':False,'logit_bias':{'154841':100}})
 else:
  body.update(messages=[{'role':'user','content':'Say hello.'}],max_tokens=1,temperature=0,skip_special_tokens=False,logit_bias={'154841':100})
 try:
  req=urllib.request.Request(endpoint,json.dumps(body).encode(),{'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=120) as response:record={'body':body,'status':response.status,'raw':response.read().decode()}
 except Exception as e:record={'body':body,'error':repr(e)}
 forced.append(record)
(out/'forced-think.json').write_text(json.dumps(forced,indent=2)+'\n')
