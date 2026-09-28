"""Recover client-only metrics without replacing an interrupted official export."""
import argparse,json,math,re,statistics
from pathlib import Path


def recover(run):
    rows=[json.loads(line) for line in (run/'c80/aiperf_artifacts/profile_export.jsonl').read_text().splitlines()]
    rows=[row for row in rows if row['metadata']['benchmark_phase']=='profiling']
    log=(run/'c80/aiperf_artifacts/logs/aiperf.log').read_text()
    done=re.search(r'Phase profiling \(profiling\) complete \| completed=([\d,]+), cancelled=([\d,]+), errors=([\d,]+)',log)
    assert done,'no client phase-completion evidence'
    completed,cancelled,errors=(int(x.replace(',','')) for x in done.groups())
    assert completed==len(rows) and errors==0 and not any(r.get('error') for r in rows)
    assert 'target: 3600.0s duration' in log and 'Phase profiling (profiling) sending complete' in log
    start=min(r['metadata']['request_start_ns'] for r in rows)
    end=max(r['metadata']['request_end_ns'] for r in rows)
    elapsed=(end-start)/1e9
    def values(key):return [r['metrics'][key]['value'] for r in rows if key in r['metrics']]
    def quantile(xs,p):
        xs=sorted(xs);i=(len(xs)-1)*p;j=int(i)
        return xs[j]+(xs[min(j+1,len(xs)-1)]-xs[j])*(i-j)
    output=sum(values('output_token_count'));throughput=output/elapsed
    env=dict(line.split('=',1) for line in (run/'c80/runtime.env').read_text().splitlines() if '=' in line and not line.startswith('#'))
    gpus=sum(int(env[role+'_NUM_WORKERS'])*int(env[role+'_TP']) for role in ['PREFILL','DECODE'])
    return {'source':str(run),'provenance':'Recovered client metrics, not a replacement official export or complete service validation','sending_window_complete':True,'completed':completed,'cancelled':cancelled,'runner_errors':errors,'output_tokens':output,'client_elapsed_s':elapsed,'output_tokens_per_second':throughput,'output_tokens_per_second_per_gpu':throughput/gpus,'ttft_mean_s':statistics.mean(values('time_to_first_token'))/1000,'ttft_p95_s':quantile(values('time_to_first_token'),.95)/1000,'positive_itl_mean_s':statistics.mean([x for x in values('inter_token_latency') if x>0])/1000,'limits':['Does not establish final service identity, diagnostic coverage or allocation stability.','Request records and the client phase-completion log are required; do not infer completion from record count alone.']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    reference=recover(a.reference);official=json.loads((a.reference/'c80/agentx_conc80.json').read_text())['request_metrics']
    expected={'output_tokens_per_second':official['throughput']['output']['tokens_per_second'],'ttft_mean_s':official['latency']['ttft']['mean'],'ttft_p95_s':official['latency']['ttft']['p95'],'positive_itl_mean_s':official['latency']['itl']['mean']}
    assert all(math.isclose(reference[k],v,abs_tol=.0000051,rel_tol=0) for k,v in expected.items()),(reference,expected)
    result=recover(a.run);result['reference_validation']={'source':str(a.reference),'matched_official_metrics':list(expected),'absolute_tolerance':.0000051}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
