#!/usr/bin/env python3
"""Collect a completed C80 baseline's provenance and reusable comparison evidence."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import re

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('run',type=Path)
a=p.parse_args();r=a.run
assert (r/'c80-completed.txt').exists(), 'Benchmark is incomplete'
def read(path):return json.loads(path.read_text())
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return {'sha256':h.hexdigest(),'bytes':path.stat().st_size}
env=dict(l.split('=',1) for l in (r/'c80/runtime.env').read_text().splitlines() if l and not l.startswith('#') and '=' in l)
summary=read(r/'analysis/summary.json')['points']['80']
resource_accounting=read(r/'analysis/runtime-summary.json')['accounting']
sampling_quality={}
for role in ('prefill','decode','router'):
 total=resource_accounting.get(f'C80/profiling/{role}/scrape_samples',0)
 failed=resource_accounting.get(f'C80/profiling/{role}/scrape_errors',0)
 sampling_quality[role]={'samples':total,'errors':failed,'error_fraction':failed/total if total else None}
lifecycle_path=r/'snapshot/engine-lifecycle-check.json'
lifecycle=read(lifecycle_path) if lifecycle_path.exists() else None
errors=collections.Counter();phases=collections.Counter()
with (r/'c80/aiperf_artifacts/profile_export.jsonl').open() as f:
 for l in f:
  x=json.loads(l);phase=x.get('metadata',{}).get('benchmark_phase','unknown');phases[phase]+=1
  if x.get('error'):errors[phase]+=1
files={}
for name in ['c80/runtime.env','c80/executed-agentx-bench.sh','c80/executed-client-patch.py','c80/benchmark_command.txt','c80/agentx_conc80.json','c80/aiperf_artifacts/profile_export.jsonl','snapshot/config/config.sh','snapshot/scripts/pin_chunk8k_dataset.py','snapshot/docker/aus_diag.py','live-containers.json','chunk-config-validation.json','smoke-validation.json','sampling/preflight.json','snapshot/resource-release.passed.json','snapshot/model-identity.json','snapshot/engine-lifecycle-check.json','analysis/joined-requests.jsonl','analysis/span-timelines.jsonl']:
 path=r/name
 if path.exists():files[name]=digest(path)
model=Path(env['MODEL'])
model_metadata={}
for name in ['config.json','tokenizer.json','tokenizer_config.json','model.safetensors.index.json']:
 if (model/name).exists():model_metadata[name]=digest(model/name)
server={role:read(r/f'launch/server-info/{role}-0.json') for role in ['prefill','decode']}
coverage=summary['phases']['profiling']['coverage']
checks={'all_exported_profile_requests_paired':coverage['paired']==coverage['client_records'] and coverage['client_records']>0,
 'model_metadata_unchanged':read(r/'snapshot/model-identity.json')==model_metadata,
 'chunk_4k_both_roles':all(x['chunked_prefill_size']==4096 for x in server.values()),
 'config_validation':read(r/'chunk-config-validation.json')['passed'],
 'resource_release':read(r/'snapshot/resource-release.passed.json')['passed'],
 'smoke_8_pairs':read(r/'smoke-validation.json')['paired_requests']==8,
 'sampler_preflight':read(r/'sampling/preflight.json')['passed'],
 'runtime_validation':read(r/'c80/baseline-validation.json')['passed'],
 'concurrency_duration_warmup':all(env[k]==v for k,v in {'CONC':'80','DURATION':'3600','AIPERF_WARMUP_REQUESTS_PER_LANE':'10','SIMULATE_ACC_LEN':'3.61'}.items())}
result={'schema_version':1,'generated_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'run':str(r),
 'collection_checks':checks,'collection_complete':all(checks.values()),'runtime_env':env,'server_info':server,
 'dataset':read(r/'c80/baseline-validation.json'),'file_manifest':files,'model_metadata':model_metadata,
 'runner_accounting':summary['runner_accounting'],'exported_records_by_phase':dict(phases),'exported_errors_by_phase':dict(errors),
 'profiling_coverage':coverage,'profiling_sampling_quality':sampling_quality,'engine_lifecycle_audit':lifecycle,
 'limitations':['Model metadata hashes do not hash every weight shard.','One baseline does not establish run-to-run variability.','Resource scrape failures are missing observations, not zero values; inspect recorded sampling error fractions.','Reuse requires same nodes/configuration and matched cache initialization/warmup; collect adjacent controls after material environment changes.','Profile errors and runner errors have different accounting; both are retained.']}
(r/'analysis/baseline-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'collection_complete':result['collection_complete'],'checks':checks,'exported_errors_by_phase':dict(errors)}))
if not result['collection_complete']:raise SystemExit(1)
