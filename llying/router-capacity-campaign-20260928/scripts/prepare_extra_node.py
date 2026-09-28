"""Prepare an allocated third node without registering it in the live benchmark pool."""
import argparse,datetime,getpass,json,re,shlex,socket,subprocess,time
from pathlib import Path

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts']
IMAGE='infera-sglang:aus-campaign-radix-20260928';BASE='infera-sglang:aus-0922-reqtrace'
BASE_ID='sha256:4f125ff9096f75611fa24caad4c01c3707dd0892251680a6483b99ffaa2a88bb'
p=argparse.ArgumentParser();p.add_argument('--job',type=int,required=True);p.add_argument('--isolated-smoke',action='store_true');a=p.parse_args()
report={'job':a.job,'state':'PENDING'}


def save(state,**fields):
    report.update(state=state,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
    target=ROOT/'extra-node.json';tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(target)
    print(state,fields,flush=True)


def remote(node,args,timeout=1800):
    return subprocess.check_output(SSH+[node,shlex.join([str(x) for x in args])],text=True,stderr=subprocess.STDOUT,timeout=timeout)


def gpu_state(node):
    return json.loads(remote(node,['rocm-smi','--showmeminfo','vram','--json'],timeout=30))


try:
    save('PENDING')
    while True:
        result=subprocess.run(['scontrol','show','job',str(a.job),'-o'],text=True,capture_output=True,timeout=15)
        if result.returncode:raise RuntimeError(result.stderr)
        state=re.search(r'\bJobState=(\S+)',result.stdout)[1]
        if state=='RUNNING':break
        if state not in ('PENDING','CONFIGURING'):raise RuntimeError(f'job {a.job}: {state}')
        time.sleep(30)
    owner=re.search(r'\bUserId=([^ (]+)',result.stdout)[1];assert owner==getpass.getuser()
    nodelist=re.search(r'\bNodeList=(\S+)',result.stdout)[1]
    nodes=subprocess.check_output(['scontrol','show','hostnames',nodelist],text=True).splitlines();assert len(nodes)==1
    node=nodes[0]
    active=json.loads((ROOT/'allocation-registry.json').read_text())['jobs']
    forbidden=set(json.loads((ROOT/'config/avoid-nodes.json').read_text()))|{n for j in active if int(j['id'])!=a.job for n in j['nodes']}
    assert node not in forbidden
    ip=socket.gethostbyname(node);save('ALLOCATED',node=node,ip=ip)
    registry=ROOT/'allocation-registry.json';data=json.loads(registry.read_text())
    data['jobs']=[j for j in data['jobs'] if j['id']!=a.job]+[{'id':a.job,'nodes':[node]}]
    tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(registry)
    try:base_id=remote(node,['docker','image','inspect','--format','{{.Id}}',BASE],timeout=30).strip()
    except subprocess.CalledProcessError:
        save('LOADING_BASE_IMAGE')
        log=remote(node,['docker','load','-i','/perf_apps/liyingli/bench_agentx/p8d8-tracing-aus-20260922/diagnostic-image.tar'])
        (ROOT/f'extra-{a.job}-image-load.log').write_text(log)
        base_id=remote(node,['docker','image','inspect','--format','{{.Id}}',BASE],timeout=30).strip()
    assert base_id==BASE_ID,(base_id,BASE_ID)
    save('BUILDING_OVERLAY')
    log=remote(node,['docker','build','-t',IMAGE,str(ROOT/'build')]);(ROOT/f'extra-{a.job}-image-build.log').write_text(log)
    image_id=remote(node,['docker','image','inspect','--format','{{.Id}}',IMAGE],timeout=30).strip();save('IMAGE_READY',image_id=image_id)
    cache=ROOT/'artifacts/aiter-prefill-cache.tar'
    if cache.exists():
        dest='/tmp/aiter-jit-100078/'+image_id.split(':')[-1]
        try:remote(node,['test','-e',dest],timeout=30)
        except subprocess.CalledProcessError:
            remote(node,['mkdir','-p',dest]);remote(node,['tar','--no-same-owner','-xf',str(cache),'-C',dest])
        save('COMPILED_CACHE_READY')
    for _ in range(120):
        gpu=gpu_state(node)
        if len([k for k in gpu if k.startswith('card')])==8 and all(int(gpu[f'card{i}']['VRAM Total Used Memory (B)'])/int(gpu[f'card{i}']['VRAM Total Memory (B)'])<.02 for i in range(8)):break
        time.sleep(5)
    else:raise RuntimeError('allocated node GPUs remain occupied; no foreign process was stopped')
    do_smoke=False
    current=Path('/home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928/CURRENT.json')
    if a.isolated_smoke and current.exists():
        state=json.loads(current.read_text())
        window=state.get('profile_window_utc',[])
        if state.get('stage','').endswith('PROFILING') and len(window)==2:
            end=datetime.datetime.fromisoformat(window[1].replace('Z','+00:00'))
            do_smoke=(end-datetime.datetime.now(datetime.timezone.utc)).total_seconds()>600
    if do_smoke:
        prefix=f'llying-campaign-extra-{a.job}';run=ROOT/'runs'/f'extra-prep-{a.job}';run.mkdir(parents=True,exist_ok=True)
        config=ROOT/'config'/f'extra-prep-{a.job}.sh'
        config.write_text(f'''source {ROOT}/config/b1-smoke.sh
    export RUN_ID=extra-prep-{a.job}
    export RUN={run}
    export CONTAINER_PREFIX={prefix}
    export ALLOCATION_JOB_ID={a.job}
    export PREFILL_NODE={node}
    export PREFILL_IP={ip}
    export PREFILL_EXTRA_ARGS="--random-seed 823508857"
    export PREFILL_HICACHE=0
    ''')
        etcd=prefix+'-etcd';worker=prefix+'-prefill-0'
        remote(node,['docker','run','-d','--init','--name',etcd,'--network','host','--label','infera.allocation-job='+str(a.job),'quay.io/coreos/etcd:v3.5.14','etcd','--advertise-client-urls',f'http://{ip}:23379','--listen-client-urls','http://0.0.0.0:23379','--listen-peer-urls','http://127.0.0.1:23380','--initial-advertise-peer-urls','http://127.0.0.1:23380','--initial-cluster','default=http://127.0.0.1:23380'])
        save('ISOLATED_PREFILL_STARTING')
        remote(node,['bash',str(ROOT/'scripts/bench-harness/engine.sh'),'prefill','prefill-0',ip,'0,1,2,3,4,5,6,7',29101,29098,25657,28901,worker,f'{ip}:23379','CONFIG='+str(config),'SERVER_LOG='+str(run/'prefill.log')])
        subprocess.run(['python3',str(ROOT/'scripts/bench-harness/tools/wait_healthy.py'),'--target','prefill',node,worker,f'http://{ip}:29101/health','--ssh-options','-F /dev/null -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/tmp/bench-agentx-known-hosts','--timeout','1800','--interval','10','--summary',str(run/'health.json')],check=True,timeout=1900)
        save('ISOLATED_PREFILL_PASSED')
        remote(node,['docker','stop','-t','60',worker,etcd],timeout=150)
        for _ in range(120):
            gpu=gpu_state(node)
            if all(int(gpu[f'card{i}']['VRAM Total Used Memory (B)'])/int(gpu[f'card{i}']['VRAM Total Memory (B)'])<.02 for i in range(8)):break
            time.sleep(5)
        else:raise RuntimeError('isolated prefill VRAM release timed out')
    save('PREPARED',gpu_memory=gpu_state(node),isolated_smoke=do_smoke)
except BaseException as exc:
    save('FAILED',error=str(exc))
    raise
