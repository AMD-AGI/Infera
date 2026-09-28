"""Prepare replacement allocations after externally preempted measurements."""
import concurrent.futures,datetime,getpass,json,os,re,shlex,socket,subprocess,time
from pathlib import Path

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928');JOB=32053
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts']
IMAGE='infera-sglang:aus-campaign-radix-20260928';BASE='infera-sglang:aus-0922-reqtrace'
BASE_ID='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
raw=subprocess.check_output(['scontrol','show','job',str(JOB),'-o'],text=True)
assert re.search(r'\bUserId=([^ (]+)',raw)[1]==getpass.getuser()
assert 'JobState=RUNNING' in raw
nodes=subprocess.check_output(['scontrol','show','hostnames',re.search(r'\bNodeList=(\S+)',raw)[1]],text=True).splitlines();assert len(nodes)==2
assert not set(nodes)&{'smci355-ccs-aus-n04-29','smci355-ccs-aus-n01-25'}
pnode='smci355-ccs-aus-n10-29' if 'smci355-ccs-aus-n10-29' in nodes else nodes[0]
dnode=next(n for n in nodes if n!=pnode)
report={'job':JOB,'state':'PREPARING','nodes':nodes,'prefill_node':pnode,'decode_node':dnode,'prefill_ip':socket.gethostbyname(pnode),'decode_ip':socket.gethostbyname(dnode)}
registry=ROOT/'allocation-registry.json';tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps({'jobs':[{'id':JOB,'nodes':nodes}]})+'\n');tmp.replace(registry)


def save(state,**fields):
    report.update(state=state,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    p=ROOT/'recovery-allocation.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
    print(state,fields,flush=True)


def remote(node,args,timeout=1800):
    return subprocess.check_output(SSH+[node,shlex.join(list(map(str,args)))],text=True,stderr=subprocess.STDOUT,timeout=timeout)


def prepare(node):
    # These names were recorded before the old allocation was preempted.
    old={'llying-campaign-b1-prefill-0','llying-campaign-b3-router','llying-campaign-b3-collector','llying-campaign-b1-smoke-etcd'}
    running=remote(node,['docker','ps','--format','{{.Names}}'],timeout=30).splitlines()
    for name in sorted(old&set(running)):
        remote(node,['docker','stop','-t','60',name],timeout=120)
    try:base=remote(node,['docker','image','inspect','--format','{{.Id}}',BASE],timeout=30).strip()
    except subprocess.CalledProcessError:
        log=remote(node,['docker','load','-i','/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/diagnostic-image.tar'])
        (ROOT/f'recovery-{node}-image-load.log').write_text(log)
        base=remote(node,['docker','image','inspect','--format','{{.Id}}',BASE],timeout=30).strip()
    assert base==BASE_ID,(node,base)
    try:image_id=remote(node,['docker','image','inspect','--format','{{.Id}}',IMAGE],timeout=30).strip()
    except subprocess.CalledProcessError:
        log=remote(node,['docker','build','-t',IMAGE,str(ROOT/'build')]);(ROOT/f'recovery-{node}-image-build.log').write_text(log)
        image_id=remote(node,['docker','image','inspect','--format','{{.Id}}',IMAGE],timeout=30).strip()
    cache=ROOT/'artifacts/aiter-prefill-cache.tar';dest=f'/tmp/aiter-jit-{os.getuid()}/'+image_id.split(':')[-1]
    try:remote(node,['test','-e',dest],timeout=30)
    except subprocess.CalledProcessError:
        remote(node,['mkdir','-p',dest]);remote(node,['tar','--no-same-owner','-xf',str(cache),'-C',dest])
    for _ in range(240):
        gpu=json.loads(remote(node,['rocm-smi','--showmeminfo','vram','--json'],timeout=30))
        if all(int(gpu[f'card{i}']['VRAM Total Used Memory (B)'])/int(gpu[f'card{i}']['VRAM Total Memory (B)'])<.02 for i in range(8)):break
        time.sleep(5)
    else:raise RuntimeError(f'{node}: VRAM remains occupied; foreign processes preserved')
    return {'node':node,'image_id':image_id,'gpu_memory':gpu}


save('PREPARING')
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(prepare,nodes))
    save('PREPARED',prepared_nodes=results)
except BaseException as exc:
    save('FAILED',error=str(exc));raise
