"""Analyze the completed B3 point without changing the live service."""
import concurrent.futures,json,subprocess,time
from pathlib import Path

ROOT=Path('/perf_apps/liyingli/bench_agentx/router-capacity-20260928')
run=ROOT/'runs/campaign-b3-decode-affinity';b2=ROOT/'runs/campaign-b2-radix'
baseline=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-performance')
out=Path('/home/liyingli/code/bench_agentx/Infera/llying/router-capacity-campaign-20260928/b3')
while True:
    status=(run/'STATUS').read_text()
    if 'COMPLETE_REVIEW_PENDING' in status:break
    if 'NEEDS_REVIEW' in status:raise RuntimeError(status)
    time.sleep(30)


def call(script,*args):
    subprocess.run(['python3',str(ROOT/'scripts'/script),*map(str,args)],check=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
    jobs=[pool.submit(call,'analyze_balance.py',run),
          pool.submit(call,'compare_matched_requests.py',baseline,run,'--input-tolerance','8','--output',run/'analysis/matched-baseline'),
          pool.submit(call,'compare_matched_requests.py',b2,run,'--input-tolerance','8','--output',run/'analysis/matched-b2')]
    for job in concurrent.futures.as_completed(jobs):job.result()
call('compare_c80_runs.py',b2,run,'--reference-label','B2_D_radix','--candidate-label','B3_D_affinity','--output-dir',run/'analysis/comparison-b2')
call('archive_point.py',run,out)
assert json.loads((out/'REVIEW.json').read_text())['checks_passed']
import shutil
for source,target in [('comparison-b2/comparison.json','comparison-b2.json'),('comparison-b2/COMPARISON.zh-CN.md','COMPARISON-B2.zh-CN.md'),('matched-b2/matched-requests.json','matched-b2.json'),('matched-b2/MATCHED-REQUESTS.zh-CN.md','MATCHED-B2.zh-CN.md')]:
    shutil.copyfile(run/'analysis'/source,out/target)
(ROOT/'B3_REVIEW_READY').write_text(str(out)+'\n')
print('B3_REVIEW_READY',out,flush=True)
