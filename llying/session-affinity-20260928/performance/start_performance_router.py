import json,os,shlex,subprocess,time,urllib.request
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN']);name=os.environ['CONTAINER_PREFIX']+'-router';model=os.environ['MODEL'];node=os.environ['PREFILL_NODE']
ssh=['ssh',*shlex.split(os.environ['SSH_OPTS']),node]
def remote(args):return subprocess.check_output(ssh+[shlex.join(args)],text=True)
cmd=['docker','run','-d','--init','--name',name,'--network','host']
for key,value in {'INFERA_PD_DP_RANK_AFFINITY':'false','INFERA_PD_PREFILL_GUARD_RELEASE':'completion','INFERA_R2_DECODE_DEMAND':'off','INFERA_R3_CACHE_TIERS':'off','INFERA_R4_PREFILL_WORK':'off','INFERA_SESSION_AFFINITY':'prefill','INFERA_SESSION_AFFINITY_TTL_SECS':'3600','RUST_LOG':'info,infera_router::routing_experiments=warn'}.items():cmd+=['-e',key+'='+value]
cmd+=['-v',os.environ['ROUTER_BINARY_OVERRIDE']+':/usr/local/bin/infera-router:ro','-v',model+':'+model+':ro',os.environ['IMAGE'],'python3','-m','infera.server','--host','0.0.0.0','--port','28000','--router-backend','rust','--discovery-backend','etcd','--etcd-endpoint',os.environ['PREFILL_IP']+':22379','--request-transport','http','--kv-event-transport','zmq','--router-tokenizer-path',model,'--router-policy','kv-aware','--kv-prefill-overlap-weight','20','--kv-decode-overlap-weight','2']
(run/'snapshot/router-launch-command.json').write_text(json.dumps(cmd,indent=2));print(remote(cmd),flush=True)
