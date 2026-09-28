"""Wait for SSH admission after Slurm reports a fresh allocation as running."""
import re,subprocess,time


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
