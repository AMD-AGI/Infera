#!/usr/bin/env python3
"""Copy reviewable evidence from a completed audited case; hash large shared artifacts."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();r=a.run;out=a.output
assert 'COMPLETE_REVIEW_PENDING' in (r/'STATUS').read_text()
assert json.loads((r/'analysis/case-audit.json').read_text())['passed']
assert not (r/'INVALID').exists()
out.mkdir(parents=True,exist_ok=True)
for name in ['summary.json','span-summary.json','chunk-evidence.json','case-audit.json','guard-lifecycle-summary.json','REPORT.zh-CN.md','router-picks.json']:
 src=r/'analysis'/name
 if src.exists():shutil.copy2(src,out/name)
for name in ['c80/agentx_conc80.json','case-validation.json','snapshot/cache-empty-before-warmup.json','snapshot/config.sh','snapshot/model-identity.json','snapshot/reuse-reset-validation.json','snapshot/diagnostic-cursors.json','snapshot/capture-start-epoch.txt','c80-started.txt','c80-completed.txt','STATUS']:
 src=r/name
 if src.exists():shutil.copy2(src,out/src.name)
mode=json.loads((r/'snapshot/router-mode-validation.json').read_text());c=mode['container']
(out/'router-identity.json').write_text(json.dumps({'passed':mode['passed'],'mode':mode['mode'],'binary_sha256':mode['binary_sha256'],'container':{k:c[k] for k in ['Id','Created','Image','State','RestartCount']},'smoke_completion_log_present':mode['smoke_completion_log_present']},indent=2)+'\n')
manifest={}
for name in ['analysis/joined-requests.jsonl','analysis/span-timelines.jsonl','analysis/runtime-summary.json','analysis/guard-lifecycle-joined.jsonl','c80/aiperf_artifacts/profile_export.jsonl','server-logs/router.log','sampling/gpu-ownership.jsonl']:
 src=r/name;h=hashlib.sha256()
 with src.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 manifest[name]={'path':str(src),'bytes':src.stat().st_size,'sha256':h.hexdigest()}
(out/'large-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(out/'archive-provenance.json').write_text(json.dumps({'source':str(r),'archived_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'completed_and_audited','large_artifacts':'Retained at shared paths listed in manifest; not copied into git.'},indent=2)+'\n')
print(out)
