#!/usr/bin/env python3
import datetime,json,os,shlex,subprocess,time
from pathlib import Path
root=Path(os.environ['TRACE_RUNTIME'])
for role in ['prefill','decode']:
 p=root/'events'/('container-guard-'+role+'.jsonl')
 rows=p.read_text().splitlines();last=json.loads(rows[-1])
 assert last['event']=='heartbeat' and last['passed'],last
 stamp=datetime.datetime.fromisoformat(last['utc']).timestamp();assert time.time()-stamp<20,'Stale guard heartbeat'
 name=os.environ['CONTAINER_PREFIX']+'-watchdog-'+role
 out=subprocess.check_output(['ssh',*shlex.split(os.environ['SSH_OPTS']),os.environ[role.upper()+'_NODE'],'docker','inspect',name,'--format','{{.State.Running}}'],text=True,timeout=15)
 assert out.strip()=='true',name
print('BOTH_ALLOCATION_GUARDS_VERIFIED')
