"""Replace only changed workers across a completed case boundary."""
import concurrent.futures,json,os,re,subprocess,time,urllib.request
from pathlib import Path
import transition_two_node as ops
import launch_placement as launch

E=os.environ;RUN=Path(E['RUN']);previous=Path(E['PREVIOUS_RUN']);desired=launch.rows
status=(previous/'STATUS').read_text()
assert any(k in status for k in ['COMPLETE_REVIEW_PENDING','RADIX_GATE_PASSED','PLACEMENT_GATE_PASSED'])
command=json.loads((previous/'snapshot/router-launch-command.json').read_text())
assert command[command.index('--name')+1]==E['OLD_PREFIX']+'-router'
assert not (RUN/'c80-started.txt').exists()
if (previous/'placement-resolved.json').exists():old=json.loads((previous/'placement-resolved.json').read_text())
else:
    old=[]
    for role,port in [('prefill',29001),('decode',29002)]:
        path=previous/f'snapshot/{role}-container.json'
        if not path.exists():path=previous/f'c80/service/{role}-0-container.json'
        c=json.loads(path.read_text());env=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
        old.append({'role':role,'node':E[role.upper()+'_NODE'],'container':c['Name'].lstrip('/'),'gpu_ids':list(map(int,env['HIP_VISIBLE_DEVICES'].split(','))),'url':f"http://{E[role.upper()+'_IP']}:{port}"})
for w in desired:launch.check_allocation(w)
keep={(w['node'],w['container']) for w in desired}
retired=[w for w in old if (w['node'],w['container']) not in keep]
old_keys={(w['node'],w['container']) for w in old}
new=[w for w in desired if (w['node'],w['container']) not in old_keys]
for w in desired:
    if (w['node'],w['container']) not in old_keys:continue
    info=ops.get(w['url']+'/get_server_info')
    for key,value in {'tp_size':w['tp'],'dp_size':w['dp'],'dsa_prefill_backend':E['DSA_PREFILL_BACKEND'],'dsa_decode_backend':E['DSA_DECODE_BACKEND'],'enable_hierarchical_cache':w['role']=='prefill'}.items():assert info[key]==value,('cannot reuse changed worker',w['container'],key)
    if w['role']=='decode':
        c=json.loads(ops.remote(w['node'],['docker','inspect',w['container']]))[0];env=dict(v.split('=',1) for v in c['Config']['Env'] if '=' in v)
        assert env.get('SGLANG_SIMULATE_ACC_LEN','')==E.get('DECODE_SIMULATE_ACC_LEN','')
        assert info['disaggregation_decode_enable_radix_cache']==(E.get('DECODE_RADIX','0')=='1')
for _ in range(60):
    busy=[]
    for w in old:
        with urllib.request.urlopen(w['url']+'/metrics',timeout=10) as f:raw=f.read().decode()
        for name in ['num_running_reqs','num_queue_reqs','num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs','num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs']:
            busy.extend(float(v) for v in re.findall(r'^sglang:'+name+r'(?:\{[^}]*\})? ([0-9.eE+-]+)$',raw,re.M))
    if busy and not any(busy):break
    time.sleep(2)
else:raise RuntimeError('old workers did not drain')
for d in ['snapshot','logs','traces','sampling','launch/server-info','launch/server-logs']:(RUN/d).mkdir(parents=True,exist_ok=True)
(RUN/'snapshot/config.sh').write_bytes(Path(E['CONFIG']).read_bytes())
(RUN/'snapshot/capture-start-epoch.txt').write_text(str(int(time.time()))+'\n')
for suffix in ['router','collector']:ops.stop(E['PREFILL_NODE'],E['OLD_PREFIX']+'-'+suffix)
ops.save('OLD_CONTROL_STOPPED')
ops.remote(E['PREFILL_NODE'],['docker','run','-d','--init','--name',E['CONTAINER_PREFIX']+'-collector','--network','host','--label','infera.allocation-job='+E['ALLOCATION_JOB_ID'],'-v',E['TRACE_RUNTIME']+':'+E['TRACE_RUNTIME'],E['IMAGE'],'python3',E['TRACE_RUNTIME']+'/scripts/otlp_jsonl_collector.py','--output',str(RUN/'traces/spans.jsonl'),'--ready-file',str(RUN/'traces/ready.json')])


def change_host(node):
    removed=[w for w in retired if w['node']==node]
    starts=[w for w in new if w['node']==node]
    for w in sorted(removed,key=lambda w:w['role']!='prefill'):ops.stop(node,w['container'])
    gpus=set(i for w in removed for i in w['gpu_ids'])
    deadline=time.monotonic()+2400
    while gpus and time.monotonic()<deadline:
        data=json.loads(ops.remote(node,['rocm-smi','--showmeminfo','vram','--json']))
        if all(int(data[f'card{i}']['VRAM Total Used Memory (B)'])/int(data[f'card{i}']['VRAM Total Memory (B)'])<.02 for i in gpus):break
        time.sleep(10)
    else:
        if gpus:raise TimeoutError(f'{node}: old VRAM not released')
    if not starts:
        print('RETIRED_NODE_FREE',node,flush=True);return node
    launch.launch_host(starts)
    return node


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    nodes=sorted({w['node'] for w in retired+new})
    for future in concurrent.futures.as_completed([pool.submit(change_host,node) for node in nodes]):ops.save('NODE_READY',node=future.result())
launch.record_placement()
subprocess.run(['python3',E['TRACE_RUNTIME']+'/scripts/start_performance_router.py'],check=True)
ops.wait_healthy(E['PREFILL_NODE'],E['CONTAINER_PREFIX']+'-router','http://'+E['PREFILL_IP']+':28000')
ops.save('READY_FOR_PREFLIGHT')
