"""Archive a finished placement only after client, process and prefix checks pass."""
import argparse,datetime,json,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('archive',type=Path);p.add_argument('--wait',action='store_true');a=p.parse_args()
ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
while True:
    status=(a.run/'STATUS').read_text() if (a.run/'STATUS').exists() else ''
    if 'COMPLETE_REVIEW_PENDING' in status:break
    if not a.wait or 'NEEDS_REVIEW' in status:raise RuntimeError(status or 'run is not complete')
    time.sleep(15)
comparison=json.loads((a.run/'analysis/comparison-baseline/comparison.json').read_text())
subprocess.run(['python3',str(ROOT/'scripts/compare_matched_requests.py'),comparison['baseline'],str(a.run),'--input-tolerance','8','--output',str(a.run/'analysis/matched-baseline')],check=True)
subprocess.run(['python3',str(ROOT/'scripts/archive_point.py'),str(a.run),str(a.archive)],check=True)
review=json.loads((a.archive/'REVIEW.json').read_text());assert review['checks_passed'],review['checks']
prefix=json.loads((a.run/'analysis/decode-local-prefix.json').read_text())
assert prefix['event_coverage']==1 and prefix['completed_requests']==review['coverage']['client_records'],prefix
ready={'archive':str(a.archive),'checks_passed':True,'decode_prefix_coverage':1,'saved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(a.run/'review-ready.json').write_text(json.dumps(ready,indent=2)+'\n')
print('REVIEW_READY',a.archive,flush=True)
