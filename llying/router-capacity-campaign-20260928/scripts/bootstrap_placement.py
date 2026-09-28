"""Start the first placement on a verified, empty replacement allocation."""
import json,os,subprocess,time
from pathlib import Path
import transition_two_node as ops
import launch_placement as launch

E=os.environ;RUN=Path(E['RUN']);ROOT=Path(E['TRACE_RUNTIME'])
assert json.loads((ROOT/'recovery-allocation.json').read_text())['state']=='PREPARED'
assert not (RUN/'placement-resolved.json').exists()
for w in launch.rows:launch.check_allocation(w)
for d in ['snapshot','logs','traces','sampling','launch/server-info','launch/server-logs']:(RUN/d).mkdir(parents=True,exist_ok=True)
(RUN/'snapshot/config.sh').write_bytes(Path(E['CONFIG']).read_bytes())
(RUN/'snapshot/capture-start-epoch.txt').write_text(str(int(time.time()))+'\n')
node=E['PREFILL_NODE'];ip=E['PREFILL_IP'];prefix=E['CONTAINER_PREFIX'];label='infera.allocation-job='+E['ALLOCATION_JOB_ID']
ops.remote(node,['docker','run','-d','--init','--name',prefix+'-etcd','--network','host','--label',label,'quay.io/coreos/etcd:v3.5.14','etcd','--advertise-client-urls',f'http://{ip}:22379','--listen-client-urls','http://0.0.0.0:22379','--listen-peer-urls','http://127.0.0.1:22380','--initial-advertise-peer-urls','http://127.0.0.1:22380','--initial-cluster','default=http://127.0.0.1:22380'])
ops.remote(node,['docker','run','-d','--init','--name',prefix+'-collector','--network','host','--label',label,'-v',E['TRACE_RUNTIME']+':'+E['TRACE_RUNTIME'],E['IMAGE'],'python3',str(ROOT/'scripts/otlp_jsonl_collector.py'),'--output',str(RUN/'traces/spans.jsonl'),'--ready-file',str(RUN/'traces/ready.json')])
subprocess.run(['python3',str(ROOT/'scripts/launch_placement.py')],check=True)
subprocess.run(['python3',str(ROOT/'scripts/start_performance_router.py')],check=True)
ops.wait_healthy(node,prefix+'-router',f'http://{ip}:28000')
ops.save('READY_FOR_REAL_ACCEPTANCE_PROBE')
