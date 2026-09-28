"""Real SGLang requests; compare session assignments against P/D request summaries."""
import argparse, concurrent.futures, json, os, re, shlex, subprocess, time, urllib.request, uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--mode',choices=['both','prefill','off'],required=True);p.add_argument('--ttl',type=int,default=3600);a=p.parse_args()
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN'])
base='http://10.235.192.140:28000';tag=f'{a.mode}-{a.ttl}-{uuid.uuid4().hex[:8]}'
rows=[]
def call(index,session,stream=False,cancel=False):
 rid=f'affinity-{tag}-{index}'
 body={'rid':rid,'model':'glm5.2-mxfp4','messages':[{'role':'user','content':f'Case {session if session == "repeat" else index}. '+('The temperature is 20 degrees. '*40)+' Reply briefly with the temperature.'}],'max_tokens':1024 if cancel else 64,'temperature':0,'stream':stream}
 headers={'Content-Type':'application/json'}
 if session is not None:headers['X-Dynamo-Session-ID']=f'{tag}-{session}'
 start=time.time_ns();req=urllib.request.Request(base+'/v1/chat/completions',json.dumps(body).encode(),headers)
 with urllib.request.urlopen(req,timeout=180) as resp:
  status=resp.status
  if stream:
   lines=[]
   for line in resp:
    lines.append(line.decode())
    if cancel and line.startswith(b'data:') and b'content' in line:break
   result={'sse':''.join(lines)}
  else:result=json.load(resp)
 assert status==200
 if not stream:assert result.get('choices'),result
 return {'rid':rid,'session':session,'stream':stream,'cancel':cancel,'start_ns':start,'end_ns':time.time_ns(),'response':result}
def metrics():
 with urllib.request.urlopen(base+'/metrics',timeout=10) as f:s=f.read().decode()
 return {m.group(1):int(m.group(2)) for m in re.finditer(r'^(infera_router_session_\S+) (\d+)$',s,re.M)}
# Router subscribed after engine startup; establish a rooted event stream before probes.
with urllib.request.urlopen(urllib.request.Request('http://10.235.192.140:29001/flush_cache',data=b'',method='POST'),timeout=20) as f:
 flush_response=f.read().decode()
time.sleep(1)
start_metrics=metrics()
for i in range(3):rows.append(call(i,'repeat',stream=i==2))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 rows.extend(ex.map(lambda i:call(i,'concurrent'),range(3,7)))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 rows.extend(ex.map(lambda i:call(i,f'independent-{i}'),range(7,15)))
rows.append(call(15,None))
rows.append(call(16,'cancel',stream=True,cancel=True))
rows.append(call(17,'cancel'))
if a.ttl<30:
 time.sleep(a.ttl+1)
 rows.append(call(18,'repeat'))
for _ in range(30):
 end_metrics=metrics()
 if all(v==0 for k,v in end_metrics.items() if '_active{' in k):break
 time.sleep(1)
assert all(v==0 for k,v in end_metrics.items() if '_active{' in k),end_metrics
ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts']
def summaries(node,role):
 code=f'''import glob,json
out=[]
for f in glob.glob('/tmp/aus-diag-{os.environ.get('DIAG_SOURCE_'+role.upper(),os.environ['RUN_ID'])}/{role}/*.jsonl'):
 for line in open(f):
  try:r=json.loads(line)
  except ValueError:continue
  if r.get('event')=='request_summary' and str(r.get('rid','')).startswith('affinity-{tag}-'):out.append(r)
print(json.dumps(out))'''
 return json.loads(subprocess.check_output(ssh+[node,'python3 -c '+shlex.quote(code)],text=True))
ps=summaries('smci355-ccs-aus-n10-29','prefill');ds=summaries('smci355-ccs-aus-n03-33','decode')
(run/f'{tag}-raw.json').write_text(json.dumps({'requests':rows,'prefill':ps,'decode':ds,'metrics_before':start_metrics,'metrics_after':end_metrics},indent=2))
pmap={r['rid']:r for r in ps};dmap={r['rid']:r for r in ds}
for r in rows:
 if r['cancel']:continue
 assert r['rid'] in pmap and r['rid'] in dmap,('missing summary',r['rid'])
 assert pmap[r['rid']]['room']==dmap[r['rid']]['room'],r['rid']
for session in ['repeat','concurrent']:
 ids=[r['rid'] for r in rows if r['session']==session and not r['rid'].endswith('-18')]
 if a.mode in ['both','prefill']:assert len({pmap[i]['dp_rank'] for i in ids})==1,('P pin',session)
 if a.mode=='both':assert len({dmap[i]['dp_rank'] for i in ids})==1,('D pin',session)
if a.mode=='prefill':assert end_metrics['infera_router_session_hits_total{role="decode"}']==0
if a.mode=='off':assert all(v==0 for v in end_metrics.values())
if a.ttl<30:
 log=subprocess.check_output(ssh+['smci355-ccs-aus-n10-29','docker logs '+shlex.quote(os.environ['CONTAINER_PREFIX']+'-router')],text=True,stderr=subprocess.STDOUT)
 (run/f'{tag}-router.log').write_text(log)
 log=re.sub(r'\x1b\[[0-9;]*m','',log)
 assert re.search(r'reason=.*expired',log),'TTL did not produce an expiry selection'
 for role in ['prefill','decode']:
  assert end_metrics[f'infera_router_session_selected_total{{role="{role}"}}']>=12
log=subprocess.check_output(ssh+['smci355-ccs-aus-n10-29','docker logs '+shlex.quote(os.environ['CONTAINER_PREFIX']+'-router')],text=True,stderr=subprocess.STDOUT)
log=re.sub(r'\x1b\[[0-9;]*m','',log)
(run/f'{tag}-router.txt').write_text(log)
policy_rows=[line for line in log.splitlines() if 'pick policy="kv-aware"' in line]
assert policy_rows and all(re.search(r'picked=\S+#dp[0-7] ',line) for line in policy_rows),'policy lost DP rank'
cache_rows=[line for line in policy_rows if 'role=Prefill' in line and re.search(r'cache_hits=[1-9][0-9]* ',line)]
assert cache_rows,'no Router GPU-cache hit observed after rooted flush and repeated input'
report={'passed':True,'binary_sha256':__import__('hashlib').sha256(Path(os.environ['ROUTER_BINARY_OVERRIDE']).read_bytes()).hexdigest(),'cache_hit_observations':len(cache_rows),'flush_response':flush_response,'mode':a.mode,'ttl':a.ttl,'requests':len(rows),'cancelled_intentionally':1,'paired_completed':len(rows)-1,'assignments':[{'rid':r['rid'],'session':r['session'],'p':pmap.get(r['rid'],{}).get('dp_rank'),'d':dmap.get(r['rid'],{}).get('dp_rank')} for r in rows],'metrics':end_metrics,'raw_file':f'{tag}-raw.json'}
(run/f'{tag}-result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
