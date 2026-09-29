"""Wait for SSH admission after Slurm reports a fresh allocation as running."""
import fcntl,json,re,subprocess,time


def wait_for_ssh(remote,node,job,timeout=180):
    deadline=time.monotonic()+timeout;last_error=None
    while time.monotonic()<deadline:
        try:
            remote(node,['true'],timeout=15)
            return
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as error:
            last_error=error
            raw=subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True,timeout=15)
            state=re.search(r'\bJobState=(\S+)',raw)[1]
            if state!='RUNNING':raise RuntimeError(f'job {job} became {state}') from error
            preempt=re.search(r'\bPreemptTime=(\S+)',raw)
            if preempt and preempt[1]!='None':raise RuntimeError(f'job {job} is being preempted') from error
            time.sleep(5)
    raise TimeoutError(f'{node}: SSH admission did not become ready') from last_error


def retire_previous_containers(remote,node,expired_jobs,output):
    ids=remote(node,['docker','ps','-q','--filter','name=llying-campaign-'],timeout=30).split()
    retired=[]
    if ids:
        containers=json.loads(remote(node,['docker','inspect',*ids],timeout=30))
        for c in containers:
            job=(c['Config'].get('Labels') or {}).get('infera.allocation-job')
            if c['Name'].lstrip('/').startswith('llying-campaign-') and job in {str(x) for x in expired_jobs}:
                retired.append({'id':c['Id'],'name':c['Name'],'previous_job':job})
                remote(node,['docker','stop','-t','60',c['Id']],timeout=90)
    output.write_text(json.dumps(retired,indent=2)+'\n')


def record_allocation(path,job,nodes):
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        data=json.loads(path.read_text()) if path.exists() else {'jobs':[]}
        data['jobs']=[j for j in data['jobs'] if int(j['id'])!=int(job)]+[{'id':int(job),'nodes':nodes}]
        tmp=path.with_suffix(f'.{job}.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(path)
