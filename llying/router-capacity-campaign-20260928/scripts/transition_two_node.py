"""Transition an idle two-node case, reusing roles whose configuration is unchanged."""
import concurrent.futures,datetime,json,os,re,shlex,subprocess,time,urllib.request
from pathlib import Path

E=os.environ
RUN=Path(E['RUN']);ROOT=Path(E['TRACE_RUNTIME'])
SSH=['ssh',*shlex.split(E['SSH_OPTS'])]
actions=[]


def save(event,**details):
    actions.append(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),event=event,**details))
    p=RUN/'transition.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(actions,indent=2)+'\n');tmp.replace(p)
    print(event,details,flush=True)


def remote(node,args,timeout=120):
    return subprocess.check_output(SSH+[node,shlex.join([str(x) for x in args])],text=True,stderr=subprocess.STDOUT,timeout=timeout)


def get(url):
    with urllib.request.urlopen(url,timeout=10) as f:return json.load(f)


def stop(node,name):
    if not name.startswith('llying-campaign-'):raise ValueError('container outside this campaign')
    remote(node,['docker','stop','-t','60',name])


def wait_healthy(node,name,url):
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        try:
            with urllib.request.urlopen(url+'/health',timeout=5) as f:
                if f.status==200:return
        except Exception:pass
        state=json.loads(remote(node,['docker','inspect',name]))[0]['State']
        if not state['Running']:raise RuntimeError(f'{name} stopped during startup: {state}')
        time.sleep(10)
    raise TimeoutError(f'{name} not healthy')


def wait_idle():
    names=['num_running_reqs','num_queue_reqs','num_prefill_bootstrap_queue_reqs','num_prefill_inflight_queue_reqs','num_decode_prealloc_queue_reqs','num_decode_transfer_queue_reqs']
    for _ in range(60):
        busy=[]
        for role,port in [('PREFILL',29001),('DECODE',29002)]:
            with urllib.request.urlopen(f"http://{E[role+'_IP']}:{port}/metrics",timeout=10) as f:s=f.read().decode()
            for name in names:
                busy.extend(float(x) for x in re.findall(r'^sglang:'+name+r'(?:\{[^}]*\})? ([0-9.eE+-]+)$',s,re.M))
        if busy and not any(busy):return
        time.sleep(2)
    raise RuntimeError('previous engines did not drain; preserve them for review')


def replace_role(role):
    upper=role.upper();node=E[upper+'_NODE'];old=E['OLD_'+upper+'_CONTAINER']
    previous=json.loads(remote(node,['docker','inspect',old]))[0]
    env=dict(x.split('=',1) for x in previous['Config']['Env'] if '=' in x)
    old_gpus=[int(x) for x in env['HIP_VISIBLE_DEVICES'].split(',')]
    stop(node,old)
    deadline=time.monotonic()+2400
    while time.monotonic()<deadline:
        info=json.loads(remote(node,['rocm-smi','--showmeminfo','vram','--json']))
        fractions=[int(info[f'card{i}']['VRAM Total Used Memory (B)'])/int(info[f'card{i}']['VRAM Total Memory (B)']) for i in old_gpus]
        if all(v<.02 for v in fractions):break
        time.sleep(10)
    else:raise TimeoutError(f'{role} VRAM release timed out; no reset attempted')
    instance=role+'-0';port=29001 if role=='prefill' else 29002
    offset=0 if role=='prefill' else 1
    container=E.get(upper+'_CONTAINER',E['CONTAINER_PREFIX']+'-'+instance)
    command=['bash',E['BENCH_DIR']+'/engine.sh',role,instance,E[upper+'_IP'],E[upper+'_GPU_DEVICES'],port,28998+offset,25557+offset,28801+offset,container,E['PREFILL_IP']+':22379','CONFIG='+E['CONFIG'],'SERVER_LOG='+str(RUN/'launch/server-logs'/f'{instance}.log')]
    remote(node,command)
    wait_healthy(node,container,f"http://{E[upper+'_IP']}:{port}")
    return role,container


if __name__=='__main__':
    if (RUN/'c80-started.txt').exists():raise SystemExit('case already started; do not overwrite')
    for d in ['logs','snapshot','traces','sampling','launch/server-info','launch/server-logs']:(RUN/d).mkdir(parents=True,exist_ok=True)
    wait_idle();save('PREVIOUS_ENGINES_DRAINED')
    (RUN/'snapshot/config.sh').write_bytes(Path(E['CONFIG']).read_bytes())
    (RUN/'snapshot/capture-start-epoch.txt').write_text(str(int(time.time()))+'\n')
    old_prefix=E['OLD_PREFIX']
    for suffix in ['router','collector']:stop(E['PREFILL_NODE'],old_prefix+'-'+suffix)
    save('OLD_ROUTER_AND_COLLECTOR_STOPPED')
    name=E['CONTAINER_PREFIX']+'-collector'
    remote(E['PREFILL_NODE'],['docker','run','-d','--init','--name',name,'--network','host','-v',str(ROOT)+':'+str(ROOT),E['IMAGE'],'python3',str(ROOT/'scripts/otlp_jsonl_collector.py'),'--output',str(RUN/'traces/spans.jsonl'),'--ready-file',str(RUN/'traces/ready.json')])
    roles=[role for role in ['prefill','decode'] if E.get('RESTART_'+role.upper(),'0')=='1']
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(replace_role,role) for role in roles]
        for f in concurrent.futures.as_completed(futures):
            role,container=f.result();save('ENGINE_READY',role=role,container=container)
    subprocess.run(['python3',str(ROOT/'scripts/start_performance_router.py')],check=True)
    wait_healthy(E['PREFILL_NODE'],E['CONTAINER_PREFIX']+'-router','http://'+E['PREFILL_IP']+':28000')
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        data=get('http://'+E['PREFILL_IP']+':28000/v1/workers')
        workers=data if isinstance(data,list) else data.get('workers',data.get('data',[]))
        if len(workers)==2:break
        time.sleep(2)
    else:raise RuntimeError('expected two registered workers')
    (RUN/'launch/workers.json').write_text(json.dumps(data,indent=2)+'\n')
    live={}
    for role,port in [('prefill',29001),('decode',29002)]:
        upper=role.upper();name=E.get(upper+'_CONTAINER',E['CONTAINER_PREFIX']+'-'+role+'-0')
        c=json.loads(remote(E[upper+'_NODE'],['docker','inspect',name]))[0]
        live[role]={'Id':c['Id'],'name':name}
        info=get(f"http://{E[upper+'_IP']}:{port}/get_server_info")
        (RUN/f'launch/server-info/{role}-0.json').write_text(json.dumps(info,indent=2)+'\n')
    (RUN/'live-containers.json').write_text(json.dumps(live,indent=2)+'\n')
    save('READY_FOR_PREFLIGHT')
