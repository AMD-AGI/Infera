#!/usr/bin/env python3
import json,os,subprocess,time,urllib.request
from pathlib import Path
r=Path(os.environ['RUN']);prefix=os.environ['CONTAINER_PREFIX'];p_ip=os.environ['PREFILL_IP'];model=os.environ['MODEL']
(r/'launch/server-info').mkdir(parents=True,exist_ok=True)
def get(url):
 with urllib.request.urlopen(url,timeout=8) as f:return f.read()
for attempt in range(240):
 states={}
 for role,ip,port in [('prefill',p_ip,29001),('decode',os.environ['DECODE_IP'],29002)]:
  try:get(f'http://{ip}:{port}/health');states[role]=True
  except Exception as e:states[role]=str(e)
 if all(x is True for x in states.values()):break
 if attempt%5==0:print('WAITING_ENGINE_HEALTH',json.dumps(states),flush=True)
 time.sleep(3)
else:raise SystemExit('Engine health did not become ready')
mode=os.environ['GUARD_MODE'];assert mode in ('decode','completion')
cmd=['docker','run','-d','--init','--name',prefix+'-router','--network','host','-e','INFERA_PD_DP_RANK_AFFINITY=false','-e','INFERA_PD_PREFILL_GUARD_RELEASE='+mode,'-v',model+':'+model+':ro','-v',os.environ['ROUTER_BINARY_OVERRIDE']+':/usr/local/bin/infera-router:ro',os.environ['IMAGE'],'python3','-m','infera.server','--host','0.0.0.0','--port','28000','--router-backend','rust','--discovery-backend','etcd','--etcd-endpoint',p_ip+':22379','--request-transport','http','--kv-event-transport','zmq','--router-tokenizer-path',model,'--router-policy','kv-aware','--kv-prefill-overlap-weight',os.environ['KV_PREFILL_OVERLAP_WEIGHT'],'--kv-decode-overlap-weight',os.environ['KV_DECODE_OVERLAP_WEIGHT']]
(r/'snapshot/router-launch-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
subprocess.run(cmd,check=True)
for i in range(100):
 try:get(f'http://{p_ip}:28000/health');break
 except Exception:time.sleep(3)
else:raise SystemExit('Router health timeout')
workers=get(f'http://{p_ip}:28000/v1/workers');(r/'launch/workers.json').write_bytes(workers)
x=json.loads(workers);ws=x if isinstance(x,list) else x.get('workers') or x.get('data') or x.get('instances')
assert len(ws)==2,ws
for role,ip,port in [('prefill',p_ip,29001),('decode',os.environ['DECODE_IP'],29002)]:
 (r/f'launch/server-info/{role}-0.json').write_bytes(get(f'http://{ip}:{port}/get_server_info'))
print('ROUTER_READY',mode,flush=True)
