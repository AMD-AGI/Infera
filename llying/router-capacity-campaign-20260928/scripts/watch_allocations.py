"""Stop only campaign containers after confirmed lease loss or imminent expiry."""
import datetime,json,os,re,shlex,subprocess,time
from pathlib import Path

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
SSH=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','ConnectTimeout=10','-o','StrictHostKeyChecking=accept-new','-o','UserKnownHostsFile=/tmp/bench-agentx-known-hosts']
seen=set();last={}
if (ROOT/'allocation-watch.json').exists():
    seen.update(json.loads((ROOT/'allocation-watch.json').read_text()).get('seen_running',[]))


def clean(node,job):
    cmd=SSH+[node,shlex.join(['docker','ps','--format','{{.Names}} {{.Label "infera.allocation-job"}}'])]
    result=subprocess.run(cmd,text=True,capture_output=True,timeout=20)
    if result.returncode:return {'node':node,'error':result.stderr.strip()}
    names=[]
    for line in result.stdout.splitlines():
        fields=line.split();name=fields[0];label=fields[1] if len(fields)>1 else ''
        if name.startswith('llying-campaign-') and (label==job or (not label and job=='31999')):names.append(name)
    names.sort(key=lambda n:0 if '-prefill-' in n else 1 if '-decode-' in n else 2)
    if not names:return {'node':node,'containers':[]}
    command=SSH+[node,shlex.join(['docker','stop','-t','30',*names])]
    stopped=subprocess.run(command,text=True,capture_output=True,timeout=180)
    return {'node':node,'containers':names,'returncode':stopped.returncode,'stdout':stopped.stdout,'stderr':stopped.stderr}


while not (ROOT/'CAMPAIGN_COMPLETE').exists():
    observations=[]
    try:
        jobs=json.loads((ROOT/'allocation-registry.json').read_text())['jobs']
        for job in jobs:
            ident=str(job['id'])
            result=subprocess.run(['scontrol','show','job',ident,'-o'],text=True,capture_output=True,timeout=15,env=dict(os.environ,TZ='UTC'))
            state=re.search(r'\bJobState=(\S+)',result.stdout)
            row={'id':ident,'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            if result.returncode or not state:
                row.update(state='UNKNOWN',error=result.stderr.strip());observations.append(row);continue
            status=state[1];row['state']=status
            if status=='RUNNING':seen.add(ident)
            end=re.search(r'\bEndTime=(\S+)',result.stdout);left=None
            if end and end[1] not in ('Unknown','None','N/A'):
                when=datetime.datetime.fromisoformat(end[1]).replace(tzinfo=datetime.timezone.utc)
                left=(when-datetime.datetime.now(datetime.timezone.utc)).total_seconds();row['seconds_left']=left
            removed_nodes=[]
            if status=='RUNNING':
                listed=re.search(r'\bNodeList=(\S+)',result.stdout)
                if listed and listed[1] not in ('(null)','None'):
                    expanded=subprocess.run(['scontrol','show','hostnames',listed[1]],text=True,capture_output=True,timeout=15)
                    if expanded.returncode==0 and expanded.stdout.strip():
                        actual=set(expanded.stdout.splitlines());row['nodes']=sorted(actual)
                        removed_nodes=[node for node in job['nodes'] if node not in actual]
            lost=ident in seen and status!='RUNNING'
            expiring=status=='RUNNING' and left is not None and left<=300
            if lost or expiring or removed_nodes:
                row['action']='LEASE_LOST' if lost else 'LEASE_NODE_CHANGED' if removed_nodes else 'EXPIRY_BACKSTOP'
                row['cleanup']=[clean(node,ident) for node in (removed_nodes if removed_nodes and not lost and not expiring else job['nodes'])]
                if row['action']!=last.get(ident) or any(x.get('containers') for x in row['cleanup']):
                    with (ROOT/'allocation-actions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                last[ident]=row['action']
            observations.append(row)
        tmp=ROOT/'allocation-watch.tmp';tmp.write_text(json.dumps({'jobs':observations,'seen_running':sorted(seen)},indent=2)+'\n');tmp.replace(ROOT/'allocation-watch.json')
    except Exception as exc:
        with (ROOT/'allocation-watch-errors.log').open('a') as f:f.write(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} {exc}\n')
    time.sleep(30)
