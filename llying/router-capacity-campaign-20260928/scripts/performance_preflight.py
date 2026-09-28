"""Verify the reviewed configuration and a few real requests before replay."""
import hashlib,json,os,re,shlex,subprocess,time,urllib.request
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN']);url='http://'+os.environ['PREFILL_IP']+':28000';ssh=['ssh',*shlex.split(os.environ['SSH_OPTS'])]
def get(u):
 with urllib.request.urlopen(u,timeout=30) as f:return json.load(f)
infos={};live=json.loads((run/'live-containers.json').read_text())
for role,port in [('prefill',29001),('decode',29002)]:
 ip=os.environ[role.upper()+'_IP'];d=get(f'http://{ip}:{port}/get_server_info');infos[role]=d
 (run/f'launch/server-info/{role}-0.json').write_text(json.dumps(d,indent=2))
 for k,v in {'chunked_prefill_size':4096,'tp_size':8,'dp_size':8,'ep_size':1,'mem_fraction_static':0.85,'enable_hierarchical_cache':role=='prefill','random_seed':823508857 if role=='prefill' else 19197414}.items():assert d[k]==v,(role,k,d[k],v)
 if role=='prefill':
  for k,v in {'hicache_ratio':1.5,'hicache_write_policy':'write_through','hicache_io_backend':'kernel','hicache_mem_layout':'page_first'}.items():assert d[k]==v,(k,d[k])
 name=os.environ.get('PREFILL_CONTAINER',os.environ['CONTAINER_PREFIX']+'-prefill-0') if role=='prefill' else os.environ['DECODE_CONTAINER']
 c=json.loads(subprocess.check_output(ssh+[os.environ[role.upper()+'_NODE'],shlex.join(['docker','inspect',name])],text=True))[0]
 assert c['Config']['Image']==os.environ['IMAGE']
 live[role]={'Id':c['Id'],'name':name}
(run/'live-containers.json').write_text(json.dumps(live,indent=2))
for role,port in [('prefill',29001),('decode',29002)]:
 with urllib.request.urlopen(urllib.request.Request(f'http://{os.environ[role.upper()+"_IP"]}:{port}/flush_cache',data=b'',method='POST'),timeout=30) as f:(run/f'startup-flush-{role}.txt').write_bytes(f.read())
time.sleep(1)
rows=[]
for i in range(3):
 body={'rid':f'session-review-preflight-{i}','model':'glm5.2-mxfp4','messages':[{'role':'user','content':('A measurement is 20 degrees. '*100)+'Give the temperature briefly.'}],'max_tokens':64,'temperature':0,'stream':False}
 req=urllib.request.Request(url+'/v1/chat/completions',json.dumps(body).encode(),{'Content-Type':'application/json','X-Dynamo-Session-ID':'review-preflight'})
 with urllib.request.urlopen(req,timeout=180) as f:r=json.load(f)
 assert r.get('choices');rows.append(r)
with urllib.request.urlopen(url+'/metrics',timeout=10) as f:metrics=f.read().decode()
assert 'infera_router_session_hits_total{role="prefill"} 2' in metrics
assert ('infera_router_session_hits_total{role="decode"} 2' if os.environ['INFERA_SESSION_AFFINITY']=='both' else 'infera_router_session_hits_total{role="decode"} 0') in metrics
assert 'infera_router_session_active{role="prefill"} 0' in metrics
for role,port in [('prefill',29001),('decode',29002)]:
 with urllib.request.urlopen(urllib.request.Request(f'http://{os.environ[role.upper()+"_IP"]}:{port}/flush_cache',data=b'',method='POST'),timeout=30) as f:(run/f'pre-benchmark-flush-{role}.txt').write_bytes(f.read())
empty_cache={}
for role,port in [('prefill',29001),('decode',29002)]:
 for attempt in range(30):
  with urllib.request.urlopen(f'http://{os.environ[role.upper()+"_IP"]}:{port}/metrics',timeout=10) as f:raw=f.read().decode()
  values=[float(v) for v in re.findall(r'^sglang:kv_used_tokens(?:\{[^}]*\})? ([0-9.eE+-]+)$',raw,re.M)]
  evictable=[float(v) for v in re.findall(r'^sglang:kv_evictable_tokens(?:\{[^}]*\})? ([0-9.eE+-]+)$',raw,re.M)]
  if len(values)==8 and len(evictable)==8 and all(v==0 for v in values+evictable):break
  time.sleep(1)
 else:raise RuntimeError(f'{role} cache did not reach an empty state after flush: {values}')
 empty_cache[role]={'active':values,'evictable':evictable}
(run/'snapshot/cache-empty-before-warmup.json').write_text(json.dumps({'passed':True,'kv_used_tokens':empty_cache},indent=2)+'\n')
(run/'preflight.json').write_text(json.dumps({'passed':True,'requests':rows,'session_metrics':[l for l in metrics.splitlines() if l.startswith('infera_router_session_')],'binary_sha256':hashlib.sha256(Path(os.environ['ROUTER_BINARY_OVERRIDE']).read_bytes()).hexdigest()},indent=2))
print('PREFLIGHT_PASSED: 4K, HiCache1.5, three real requests, two P session hits, D unbound')
