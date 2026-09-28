import json,os,shlex,subprocess,time,urllib.request
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN']);name=os.environ['CONTAINER_PREFIX']+'-router';model=os.environ['MODEL'];node=os.environ['PREFILL_NODE']
ssh=['ssh',*shlex.split(os.environ['SSH_OPTS']),node]
def remote(args):return subprocess.check_output(ssh+[shlex.join(args)],text=True)
cmd=['docker','run','-d','--init','--name',name,'--network','host','--label','infera.campaign=router-capacity-20260928','--label','infera.allocation-job='+os.environ['ALLOCATION_JOB_ID']]
for key,value in {'INFERA_PD_DP_RANK_AFFINITY':'false','INFERA_PD_PREFILL_GUARD_RELEASE':'completion','INFERA_R2_DECODE_DEMAND':os.environ.get('INFERA_R2_DECODE_DEMAND','off'),'INFERA_R3_CACHE_TIERS':'off','INFERA_R4_PREFILL_WORK':os.environ.get('INFERA_R4_PREFILL_WORK','off'),'INFERA_P_DYNAMO_SCORE':os.environ.get('INFERA_P_DYNAMO_SCORE','off'),'INFERA_SESSION_AFFINITY':os.environ['INFERA_SESSION_AFFINITY'],'INFERA_SESSION_AFFINITY_TTL_SECS':'3600','RUST_LOG':os.environ.get('RUST_LOG','info')}.items():cmd+=['-e',key+'='+value]
cmd+=['-v',os.environ['ROUTER_BINARY_OVERRIDE']+':/usr/local/bin/infera-router:ro','-v',model+':'+model+':ro',os.environ['IMAGE'],'python3','-m','infera.server','--host','0.0.0.0','--port','28000','--router-backend','rust','--discovery-backend','etcd','--etcd-endpoint',os.environ['PREFILL_IP']+':22379','--request-transport','http','--kv-event-transport','zmq','--router-tokenizer-path',model,'--router-policy','kv-aware','--kv-prefill-overlap-weight','20','--kv-decode-overlap-weight','2']
(run/'snapshot/router-launch-command.json').write_text(json.dumps(cmd,indent=2));print(remote(cmd),flush=True)
