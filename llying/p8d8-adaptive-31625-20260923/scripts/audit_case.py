#!/usr/bin/env python3
"""Accept a completed case only with intact request, model, mode and isolation evidence."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();r=a.run
def read(path):return json.loads(path.read_text())
s=read(r/'analysis/summary.json')['points']['80'];cov=s['phases']['profiling']['coverage'];lifecycle=read(r/'analysis/guard-lifecycle-summary.json')
mode=read(r/'snapshot/router-mode-validation.json')['mode']
checks={'benchmark_finished':(r/'c80-completed.txt').exists(),'not_invalid':not (r/'INVALID').exists(),
 'case_configuration':read(r/'case-validation.json')['passed'],'cache_reset':read(r/'snapshot/cache-empty-before-warmup.json')['passed'],
 'no_profile_export_errors':cov.get('error_records',0)==0,
 'request_pairing':cov.get('paired',0)==cov.get('client_records',-1) and cov.get('paired_fraction_of_sent',0)>=.99,
 'router_mode':lifecycle['separated_observations']==(lifecycle['total_observations'] if mode=='completion' else 0),
 'completion_observation_coverage':lifecycle['coverage'].get('profiling/matched',0)>=.99*cov['client_records']}
identities={}
for role in ['prefill','decode']:
 old=read(r/f'launch/server-info/{role}-0.json')
 with urllib.request.urlopen(f"http://{old['host']}:{old['port']}/get_server_info",timeout=30) as f:new=json.load(f)
 identities[role]={'scheduler_pids_before':old.get('scheduler_pids'),'scheduler_pids_after':new.get('scheduler_pids'),'startup_time_unchanged':old.get('startup_time')==new.get('startup_time')}
 checks[role+'_unchanged']=old.get('scheduler_pids')==new.get('scheduler_pids') and identities[role]['startup_time_unchanged']
env=dict(l.split('=',1) for l in (r/'c80/runtime.env').read_text().splitlines() if l and not l.startswith('#') and '=' in l)
model=Path(env['MODEL']);before=read(r/'snapshot/model-identity.json');after={}
for name in before:
 f=model/name;h=hashlib.sha256()
 with f.open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
 after[name]={'sha256':h.hexdigest(),'bytes':f.stat().st_size}
checks['model_metadata_unchanged']=before==after
ownership=[]
with (r/'sampling/gpu-ownership.jsonl').open() as f:
 for line in f:ownership.append(json.loads(line))
checks['ownership_evidence']=bool(ownership) and not any(x['interference'] or x['consecutive_probe_errors']>=3 for x in ownership)
result={'passed':all(checks.values()),'checks':checks,'coverage':cov,'engine_identities':identities,'ownership_samples':len(ownership),'runner_accounting':s['runner_accounting'],'guard_mode':mode,'sampling_accounting':read(r/'analysis/runtime-summary.json')['accounting'],'limitations':['Single-run acceptance is not a confidence interval.','Counters with scrape errors are missing observations, not zeros.','Model metadata hashes do not hash all weight shards.']}
(r/'analysis/case-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':result['passed'],'checks':checks}))
if not result['passed']:raise SystemExit(1)
