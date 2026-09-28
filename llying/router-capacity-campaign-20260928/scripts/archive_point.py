"""Save compact, reviewable results; bulk request/trace data remain in the shared run directory."""
import argparse,datetime,hashlib,json,shutil,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run=a.run;out=a.output;out.mkdir(parents=True,exist_ok=True)
result=json.loads((run/'c80/agentx_conc80.json').read_text())
summary=json.loads((run/'analysis/summary.json').read_text());phase=summary['points']['80']['phases']['profiling']
identities={}
placement=run/'placement-resolved.json'
instances=[w['instance'] for w in json.loads(placement.read_text())] if placement.exists() else ['prefill-0','decode-0']
for instance in instances:
    before=json.loads((run/f'launch/server-info/{instance}.json').read_text())
    final=run/f'final-server-info/{instance}.json'
    if not final.exists():final=run/f"final-server-info/{instance.split('-')[0]}.json"
    after=json.loads(final.read_text())
    identities[instance]={'pids_before':before.get('scheduler_pids'),'pids_after':after.get('scheduler_pids'),'startup_unchanged':before.get('startup_time')==after.get('startup_time'),'capacity':after.get('max_total_num_tokens'),'tp':after['tp_size'],'dp':after['dp_size']}
checks={'client_completed':(run/'c80-completed.txt').exists(),'all_exported_profile_requests_paired':phase['coverage']['paired']==phase['coverage']['client_records'],'pair_rooms_match':phase['coverage'].get('room_mismatch',0)==0,'engines_unchanged':all(v['pids_before'] and v['pids_before']==v['pids_after'] and v['startup_unchanged'] for v in identities.values()),'no_exported_profile_errors':phase['coverage'].get('error_records',0)==0}
accounting=summary['points']['80'].get('runner_accounting',{})
if 'profiling' in accounting:checks['no_runner_profile_errors']=accounting['profiling'].get('errors')==0
review={'run':str(run),'saved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'checks_passed':all(checks.values()),'coverage':phase['coverage'],'runner_accounting':accounting,'engine_identities':identities,'metrics':result['request_metrics'],'note':'Checks do not prove a performance benefit; boundary cancellations and selected completed cohorts remain explicit.'}
(out/'REVIEW.json').write_text(json.dumps(review,indent=2)+'\n')
files={'c80/agentx_conc80.json':'agentx_conc80.json','c80/runtime.env':'runtime.env','c80/baseline-validation.json':'baseline-validation.json','preflight.json':'preflight.json','analysis/session-affinity.json':'session-affinity.json','analysis/guard-lifecycle-summary.json':'guard-lifecycle-summary.json','analysis/runtime-summary.json':'runtime-summary.json','analysis/decode-local-prefix.json':'decode-local-prefix.json','analysis/comparison-baseline/comparison.json':'comparison-baseline.json','analysis/comparison-baseline/COMPARISON.zh-CN.md':'COMPARISON.zh-CN.md','analysis/matched-baseline/matched-requests.json':'matched-requests.json','analysis/matched-baseline/MATCHED-REQUESTS.zh-CN.md':'MATCHED-REQUESTS.zh-CN.md','snapshot/config.sh':'config.sh'}
for source,target in files.items():
    if (run/source).exists():shutil.copyfile(run/source,out/target)
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'checks':checks,'metrics':{k:result['request_metrics']['throughput'][k] for k in ['output','per_gpu']},'ttft':result['request_metrics']['latency']['ttft'],'itl':result['request_metrics']['latency']['itl'],'cache':phase['cache']}))
