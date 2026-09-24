#!/usr/bin/env python3
import json,os,re,subprocess
from pathlib import Path
r=Path(os.environ['RUN']);c=json.loads(subprocess.check_output(['docker','inspect',os.environ['CONTAINER_PREFIX']+'-router'],text=True))[0]
env=dict(x.split('=',1) for x in c['Config']['Env'] if '=' in x)
keys=['INFERA_PD_PREFILL_GUARD_RELEASE','INFERA_R2_DECODE_DEMAND','INFERA_R3_CACHE_TIERS','INFERA_R4_PREFILL_WORK','RUST_LOG']
expected={k:os.environ[k] for k in keys}
errors=[k+' does not match reviewed configuration' for k in keys if env.get(k)!=expected[k]]
logs=subprocess.check_output(['docker','logs',c['Id']],stderr=subprocess.STDOUT,text=True)
logs=re.sub(r'\x1b\[[0-9;]*m','',logs)
if not any('router experiment controls' in x and 'decode: Off' in x and 'tiers: Off' in x and 'prefill: On' in x for x in logs.splitlines()):errors.append('Rust experiment settings not observed')
if 'routing_experiments=warn' in os.environ['RUST_LOG'] and 'routing experiment candidate' in logs:errors.append('Verbose candidate logging was not disabled')
for role,port in [('PREFILL',29001),('DECODE',29002)]:
 wid=os.environ[role+'_IP']+':'+str(port)
 if not any('render verified against this worker' in x and wid in x for x in logs.splitlines()):errors.append(role+' template parity not confirmed')
report={'passed':not errors,'errors':errors,'environment':expected,'validation_scope':'R1+R4 configuration, template parity, and logging settings; paired real requests checked by validate_smoke.py'}
(r/'experiment-smoke-validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if errors:raise SystemExit(1)
