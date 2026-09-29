"""Release a one-node allocation after a reviewed transition has emptied it."""
import argparse,datetime,fcntl,getpass,json,re,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--topology',type=Path,required=True);p.add_argument('--job',type=int,required=True);p.add_argument('--node',required=True);a=p.parse_args()
ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
while True:
    if (ROOT/'STOP_AUTORUN').exists():raise SystemExit('Stop marker set; retirement not performed')
    try:events=json.loads((a.run/'transition.json').read_text())
    except (FileNotFoundError,json.JSONDecodeError):events=[]
    if any(e['event']=='NODE_READY' and e.get('node')==a.node for e in events):break
    if (ROOT/'capacity-sequence-failed.txt').exists():raise SystemExit('Sequence failed; preserve for review')
    time.sleep(10)
rows=json.loads(a.topology.read_text());assert all(w['node']!=a.node for w in rows)
raw=subprocess.check_output(['scontrol','show','job',str(a.job),'-o'],text=True)
assert re.search(r'\bUserId=([^ (]+)',raw)[1]==getpass.getuser()
state=re.search(r'\bJobState=(\S+)',raw)[1]
if state=='RUNNING':
    hosts=subprocess.check_output(['scontrol','show','hostnames',re.search(r'\bNodeList=(\S+)',raw)[1]],text=True).splitlines()
    assert hosts==[a.node],hosts
    subprocess.run(['scancel',str(a.job)],check=True)
else:assert state in ['COMPLETING','COMPLETED','CANCELLED','PREEMPTED','TIMEOUT'],state
registry=ROOT/'allocation-registry.json'
with registry.with_suffix('.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    data=json.loads(registry.read_text());data['jobs']=[j for j in data['jobs'] if int(j['id'])!=a.job]
    tmp=registry.with_suffix('.retire.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(registry)
report={'job':a.job,'node':a.node,'released_after_gpu_free_transition':True,'previous_state':state,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(ROOT/f'released-extra-{a.job}.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
