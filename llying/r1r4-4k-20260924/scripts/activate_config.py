#!/usr/bin/env python3
import datetime,json,os,shlex,subprocess,sys
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/r1r4-4k-20260924');config=Path(sys.argv[1]).resolve()
code='set -a; source '+shlex.quote(str(config))+'; python3 -c '+shlex.quote('import os,json; print(json.dumps({k:os.environ[k] for k in ["RUN","CONTAINER_PREFIX"]}))')
v=json.loads(subprocess.check_output(['bash','-c',code],text=True))
archive=root/'events'/datetime.datetime.now(datetime.timezone.utc).strftime('previous-%Y%m%dT%H%M%SZ');archive.mkdir()
for name in ['driver.log','driver.pid','container-guard-prefill.jsonl','container-guard-decode.jsonl','cleanup-complete']:
 p=root/'events'/name
 if p.exists():p.rename(archive/name)
(root/'config/active-phase.json').write_text(json.dumps({'prefix':v['CONTAINER_PREFIX'],'run':v['RUN']},indent=2)+'\n')
(root/'active-run.txt').write_text(v['RUN']+'\n')
print(json.dumps(v))
