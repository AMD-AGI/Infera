"""Stop only this experiment's containers, P first; keep the user's allocation."""
import datetime,json,os,shlex,subprocess
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME']);run=Path(os.environ['RUN']);ssh=['ssh',*shlex.split(os.environ['SSH_OPTS'])];actions=[]
for node,names in [(os.environ['PREFILL_NODE'],[os.environ['CONTAINER_PREFIX']+'-prefill-0']),(os.environ['DECODE_NODE'],[os.environ['DECODE_CONTAINER']]),(os.environ['PREFILL_NODE'],[os.environ['CONTAINER_PREFIX']+'-router',os.environ['CONTAINER_PREFIX']+'-collector','llying-session-31999-etcd'])]:
 start=datetime.datetime.now(datetime.timezone.utc).isoformat()
 r=subprocess.run(ssh+[node,shlex.join(['docker','stop','-t','30',*names])],text=True,capture_output=True,timeout=120)
 actions.append({'node':node,'containers':names,'requested_at':start,'returned_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})
 (run/'cleanup-actions.json').write_text(json.dumps(actions,indent=2))
 print(node,names,r.returncode,flush=True)
