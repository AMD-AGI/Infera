"""Switch only the task-owned Router; leave P/D engine processes running."""
import argparse,json,shlex,subprocess,time,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['both','prefill','off']);p.add_argument('--ttl',type=int,default=3600);a=p.parse_args()
root=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928');run=root/'runs/session-affinity-31999-smoke';model='/perf_apps/data/models/GLM-5.2-MXFP4';name='llying-session-31999-router'
ssh=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts','smci355-ccs-aus-n10-29']
def remote(args):return subprocess.check_output(ssh+[shlex.join(args)],text=True,stderr=subprocess.STDOUT)
try:
 log=remote(['docker','logs',name]);(run/f'router-before-{a.mode}-{a.ttl}-{time.time_ns()}.log').write_text(log)
except subprocess.CalledProcessError:pass
remote(['docker','stop','-t','20',name]);remote(['docker','rm',name])
cmd=['docker','run','-d','--init','--name',name,'--network','host']
for k,v in {'INFERA_PD_DP_RANK_AFFINITY':'false','INFERA_PD_PREFILL_GUARD_RELEASE':'completion','INFERA_R2_DECODE_DEMAND':'off','INFERA_R3_CACHE_TIERS':'off','INFERA_R4_PREFILL_WORK':'off','INFERA_SESSION_AFFINITY':a.mode,'INFERA_SESSION_AFFINITY_TTL_SECS':str(a.ttl),'RUST_LOG':'info,infera_router::routing_experiments=warn'}.items():cmd.extend(['-e',f'{k}={v}'])
cmd+=['-v',str(root/'artifacts/infera-router-session-v2')+':/usr/local/bin/infera-router:ro','-v',model+':'+model+':ro','infera-sglang:aus-0922-reqtrace','python3','-m','infera.server','--host','0.0.0.0','--port','28000','--router-backend','rust','--discovery-backend','etcd','--etcd-endpoint','10.235.192.140:22379','--request-transport','http','--kv-event-transport','zmq','--router-tokenizer-path',model,'--router-policy','kv-aware','--kv-prefill-overlap-weight','20','--kv-decode-overlap-weight','2']
(run/f'router-{a.mode}-{a.ttl}-command.json').write_text(json.dumps(cmd,indent=2));print(remote(cmd),flush=True)
for _ in range(60):
 try:
  with urllib.request.urlopen('http://10.235.192.140:28000/v1/workers',timeout=3) as f:d=json.load(f)
  ws=d if isinstance(d,list) else d.get('workers',[])
  if len(ws)==2:print('READY',a.mode,a.ttl);break
 except Exception:pass
 time.sleep(2)
else:raise RuntimeError('Router did not become ready; models preserved')
