#!/usr/bin/env python3
"""One read-only queue check, recorded for the overnight review."""
import datetime,json,os,subprocess
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924')
os.environ['SLURM_CONF']=str(root/'config/slurm-client-dccs.conf')
p=subprocess.run(['/opt/slurm/bin/squeue','-h','-j','31719','-o','%i|%T|%R|%S'],capture_output=True,text=True,timeout=15)
row={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':p.returncode,'queue':p.stdout.strip(),'stderr':p.stderr.strip()}
with (root/'events/queue-monitor.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
print(json.dumps(row))
