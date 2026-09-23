#!/usr/bin/env python3
"""Run a gated A0/G0/A1 sequence; failures stop for review, never become gains."""
import datetime,json,os,subprocess,time
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923')
cases=['a0-guard-decode','g0-guard-completion','a1-guard-decode']
report={'cases':{},'status':'RUNNING'}
def save():
 report['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 (root/'guard-campaign-status.json').write_text(json.dumps(report,indent=2)+'\n')
def completed(case):
 r=root/'runs'/case;status=(r/'STATUS').read_text() if (r/'STATUS').exists() else ''
 if (r/'INVALID').exists() or 'FAILED' in status:raise RuntimeError(f'{case}: {status.strip()} invalid={(r/"INVALID").exists()}')
 return 'COMPLETE_REVIEW_PENDING' in status
try:
 save()
 while not completed(cases[0]):time.sleep(20)
 for i,case in enumerate(cases):
  if i:
   deadline=datetime.datetime(2026,9,24,0,43,tzinfo=datetime.timezone.utc).timestamp()
   if time.time()+7200>deadline:raise RuntimeError('Insufficient time for a complete next case before review; do not shorten it silently')
   report['status']='STARTING_'+case;save()
   env=dict(os.environ,CONFIG=str(root/'config'/(case+'.sh')))
   with (root/(case+'-driver.log')).open('w') as log:
    subprocess.run(['bash',str(root/'scripts/run_reuse_case.sh')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   if not completed(case):raise RuntimeError(f'{case}: missing completion marker')
  r=root/'runs'/case
  with (root/(case+'-guard-analysis.log')).open('w') as log:subprocess.run(['python3',str(root/'scripts/analyze_guard_lifecycle.py'),str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
  with (root/(case+'-audit.log')).open('w') as log:subprocess.run(['python3',str(root/'scripts/audit_case.py'),str(r)],stdout=log,stderr=subprocess.STDOUT,check=True)
  x=json.loads((r/'c80/agentx_conc80.json').read_text());report['cases'][case]={'completed_requests':x['num_requests_successful'],'total_tokens_s_gpu':x['request_metrics']['throughput']['per_gpu']['total_tput_tps'],'output_tokens_s':x['request_metrics']['throughput']['output']['tokens_per_second'],'ttft':x['request_metrics']['latency']['ttft'],'request_accounting':x['request_accounting']};report['status']='COMPLETED_'+case;save()
  if case=='g0-guard-completion':
   report['status']='G0_COMPLETE_AWAITING_ANALYSIS_DECISION';save()
   decision=root/'guard-decision.json'
   while not decision.exists():
    if time.time()>datetime.datetime(2026,9,24,0,43,tzinfo=datetime.timezone.utc).timestamp():raise RuntimeError('Review deadline reached while awaiting experiment decision')
    time.sleep(20)
   action=json.loads(decision.read_text())['action']
   if action=='stop':
    report['status']='GUARD_BRANCH_STOPPED_BY_ANALYSIS';save();raise SystemExit(0)
   if action!='run_a1':raise RuntimeError('Unknown analysis decision')
 report['status']='A0_G0_A1_COMPLETE_REVIEW_REQUIRED';save()
except Exception as e:
 report['status']='STOPPED_FOR_REVIEW';report['error']=str(e);save();raise
