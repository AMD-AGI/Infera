"""Launch selected explicit workers, serially per host and concurrently across hosts."""
import concurrent.futures,getpass,json,os,subprocess,sys
from pathlib import Path
import transition_two_node as ops
from gpu_inventory import Inventory
sys.path.insert(0,str(Path(__file__).parent/'bench-harness/tools'))
from campaign_topology import load

E=os.environ;RUN=Path(E['RUN']);rows=load(E['TOPOLOGY'])
selected=set(filter(None,E.get('LAUNCH_INSTANCES',','.join(r['instance'] for r in rows)).split(',')))
assert selected<=set(r['instance'] for r in rows)
_inventory=Inventory()


def gpu_memory(node,job):
    return _inventory.memory(ops.remote,node,E['IMAGE'],job,RUN/'snapshot')


def check_allocation(row):
    job=str(row['allocation_job'])
    owner=subprocess.check_output(['squeue','-h','-j',job,'-o','%u'],text=True).strip()
    assert owner==getpass.getuser(),(job,owner)
    hosts=subprocess.check_output(['squeue','-h','-j',job,'-o','%N'],text=True).strip()
    nodes=subprocess.check_output(['scontrol','show','hostnames',hosts],text=True).splitlines()
    assert row['node'] in nodes,(row['node'],nodes)


def launch_host(workers):
    for w in workers:
        check_allocation(w)
        data=gpu_memory(w['node'],w['allocation_job'])
        assert all(int(data[i]['VRAM Total Used Memory (B)'])/int(data[i]['VRAM Total Memory (B)'])<.02 for i in w['gpu_ids']),f"{w['instance']} GPUs are occupied"
        configured=json.loads(E['RDMA_DEVICE'])
        mapping={str(local):configured[str(physical)] for local,physical in enumerate(w['gpu_ids'])}
        cmd=['bash',E['BENCH_DIR']+'/engine.sh',w['role'],w['instance'],w['ip'],','.join(map(str,w['gpu_ids'])),w['engine_port'],w['bootstrap_port'],w['kv_port'],w['snapshot_port'],w['container'],E['PREFILL_IP']+':22379','CONFIG='+E['CONFIG'],'WORKER_RDMA_DEVICE='+json.dumps(mapping,separators=(',',':')),'DIAG_INSTANCE='+w['instance'],'WORKER_ALLOCATION_JOB_ID='+str(w['allocation_job']),'SERVER_LOG='+str(RUN/'launch/server-logs'/f"{w['instance']}.log")]
        print('START',w['instance'],w['node'],w['gpu_ids'],flush=True)
        ops.remote(w['node'],cmd)
        ops.wait_healthy(w['node'],w['container'],w['url'])
        print('READY',w['instance'],flush=True)


def record_placement():
    resolved=[]
    for w in rows:
        check_allocation(w)
        c=json.loads(ops.remote(w['node'],['docker','inspect',w['container']]))[0]
        info=ops.get(w['url']+'/get_server_info')
        assert info['tp_size']==w['tp'] and info['dp_size']==w['dp']
        assert info['chunked_prefill_size']==4096
        (RUN/f"launch/server-info/{w['instance']}.json").write_text(json.dumps(info,indent=2)+'\n')
        (RUN/f"snapshot/{w['instance']}-container.json").write_text(json.dumps(c,indent=2)+'\n')
        diag=next(m['Source'] for m in c['Mounts'] if m['Destination']=='/aus-diag')
        resolved.append(dict(w,container_id=c['Id'],diag_dir=diag,scheduler_pids=info.get('scheduler_pids')))
    (RUN/'placement-resolved.json').write_text(json.dumps(resolved,indent=2)+'\n')


if __name__=='__main__':
    for d in ['launch/server-info','launch/server-logs','snapshot']:(RUN/d).mkdir(parents=True,exist_ok=True)
    by_host={}
    for w in rows:
        if w['instance'] in selected:by_host.setdefault(w['node'],[]).append(w)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,len(by_host))) as pool:
        for f in concurrent.futures.as_completed([pool.submit(launch_host,ws) for ws in by_host.values()]):f.result()
    record_placement()
