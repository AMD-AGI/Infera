#!/usr/bin/env python3
import hashlib,json,os,subprocess
from pathlib import Path
r=Path(os.environ['RUN']);name=os.environ['CONTAINER_PREFIX']+'-router'
c=json.loads(subprocess.check_output(['docker','inspect',name],text=True))[0]
env=dict(x.split('=',1) for x in c['Config']['Env'] if '=' in x)
mode=os.environ['GUARD_MODE'];assert mode in ('decode','completion')
assert env.get('INFERA_PD_PREFILL_GUARD_RELEASE')==mode
logs=subprocess.check_output(['docker','logs','--tail','500',name],stderr=subprocess.STDOUT,text=True)
assert 'P/D streaming guard experiment' in logs,'Streaming experiment code was not reached'
assert 'guard-stream-smoke' in logs and 'prefill HTTP leg drained; guard lifecycle observed' in logs,'Prefill completion observation missing'
p=Path(os.environ['ROUTER_BINARY_OVERRIDE']);sha=hashlib.sha256(p.read_bytes()).hexdigest()
assert sha in (r/'snapshot/router-binary.sha256').read_text()
(r/'snapshot/router-mode-validation.json').write_text(json.dumps({'passed':True,'mode':mode,'binary_sha256':sha,'container':c,'smoke_completion_log_present':True},indent=2)+'\n')
print('ROUTER_MODE_VERIFIED',mode,sha)
