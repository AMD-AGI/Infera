"""Same-duration completed-request window; preliminary, not a final throughput estimate."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--seconds',type=int,required=True);a=p.parse_args()
def summarize(run):
 path=next((run/'c80').rglob('profile_export.jsonl'));rows=[]
 for line in path.open():
  if not line.endswith('\n'):break
  r=json.loads(line)
  if r['metadata']['benchmark_phase']=='profiling':rows.append(r)
 start=min(r['metadata']['request_start_ns'] for r in rows);end=start+a.seconds*10**9
 if max(r['metadata']['request_end_ns'] for r in rows)<end:raise SystemExit('Requested window is not complete yet')
 chosen=[r for r in rows if start<=r['metadata']['request_start_ns'] and r['metadata']['request_end_ns']<=end and not r.get('error') and not r['metadata'].get('was_cancelled')]
 def vals(key):return [r['metrics'][key]['value'] for r in chosen if key in r['metrics']]
 return {'requests':len(chosen),'output_tps_gpu':sum(vals('output_token_count'))/a.seconds/16,'ttft_mean_ms':statistics.fmean(vals('time_to_first_token')),'output_tokens_mean':statistics.fmean(vals('output_token_count')),'first_request_ns':start}
base=Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/g0-guard-completion');d={'window_seconds':a.seconds,'G0':summarize(base),'session_prefill':summarize(a.run)}
(a.run/'analysis').mkdir(exist_ok=True);(a.run/f'analysis/prefix-window-{a.seconds}.json').write_text(json.dumps(d,indent=2));print(json.dumps(d))
