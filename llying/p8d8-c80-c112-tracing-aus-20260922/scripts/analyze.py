#!/usr/bin/env python3
"""Join client cohorts with P/D diagnostics and quantify stages/rank imbalance.

Durations use same-process monotonic clocks. Cross-host ordering is intentionally
not inferred here. Samples without completion remain in the coverage denominator.
"""
import argparse
import collections
import datetime as dt
import json
import math
import statistics
from pathlib import Path

def jsonl(path):
    if not path.exists():
        return
    with path.open() as f:
        for i, line in enumerate(f, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if not line.endswith('\n'):
                    break  # concurrent snapshot can end with a partial last line
                raise ValueError(f'{path}:{i}: invalid JSON')

def stats(values):
    values=sorted(v for v in values if v is not None and math.isfinite(v))
    def q(frac):
        if not values:return None
        p=(len(values)-1)*frac; a=int(p); b=math.ceil(p)
        return values[a]+(values[b]-values[a])*(p-a)
    return dict(n=len(values),mean=statistics.fmean(values) if values else None,
                p50=q(.5),p90=q(.9),p99=q(.99),max=max(values) if values else None)

def canonical(rid):
    return str(rid or '').split('_')[-1]

def durations(row):
    times=row['times']
    def diff(end,start):
        a,b=times.get(end,0),times.get(start,0)
        return (a-b)*1000 if a>0 and b>0 and a>=b else None
    output={'queue_ms':diff('forward_entry_time','wait_queue_entry_time')}
    if row['role']=='prefill':
        output.update(bootstrap_ms=diff('wait_queue_entry_time','prefill_bootstrap_queue_entry_time'),
                      forward_envelope_ms=diff('prefill_finished_time','forward_entry_time'),
                      transfer_tail_ms=diff('prefill_kv_transfer_finish_time','prefill_transfer_queue_entry_time'))
    else:
        output.update(bootstrap_ms=diff('bootstrap_done_time','decode_prealloc_queue_entry_time'),
                      alloc_wait_ms=diff('decode_transfer_queue_entry_time','bootstrap_done_time'),
                      transfer_wait_ms=diff('wait_queue_entry_time','decode_transfer_queue_entry_time'),
                      generation_ms=diff('completion_time','forward_entry_time'))
    return output

def bin_tokens(n):
    for limit in (8192,32768,65536,131072,262144,524288):
        if n<limit:return f'lt{limit}'
    return 'ge524288'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args(); root=args.run
    out=root/'analysis';out.mkdir(exist_ok=True)
    events=[]; summaries=collections.defaultdict(list)
    request_events=collections.defaultdict(list)
    for path in sorted((root/'diagnostics').glob('*/*.jsonl')):
        for row in jsonl(path):
            events.append(row)
            if row.get('rid'):
                request_events[(canonical(row['rid']),row['role'])].append(row)
            if row.get('event')=='request_summary':
                summaries[(canonical(row['rid']),row['role'])].append(row)
    result={'generated_at':dt.datetime.now(dt.timezone.utc).isoformat(),'points':{},
            'diagnostic_events':len(events),'limitations':[
                'Tracing overhead is not established by comparison with historical runs.',
                'Host acknowledgement is CPU observation, not pure DMA duration.',
                'Cross-host clocks require independent uncertainty bounds.',
                'Prefill and Decode waits overlap; do not add their percentiles.',
                'Long-window balanced token counts do not exclude transient imbalance.']}
    joined_path=out/'joined-requests.jsonl'
    with joined_path.open('w') as joined:
        for c in (80,112):
            point=root/f'c{c}'
            if not point.exists():continue
            records=list(jsonl(point/'aiperf_artifacts/profile_export.jsonl'))
            point_result={'raw_records':len(records),'phases':{}}
            aggregate=point/f'agentx_conc{c}.json'
            if aggregate.exists():
                data=json.loads(aggregate.read_text())
                point_result['headline']={k:data.get(k) for k in ('request_accounting','request_metrics','server_metrics')}
            for phase in ('warmup','profiling'):
                cohort=[r for r in records if r.get('metadata',{}).get('benchmark_phase')==phase]
                if not cohort:continue
                counts=collections.Counter(); stage=collections.defaultdict(list)
                rank_stage=collections.defaultdict(list); stratified=collections.defaultdict(list)
                window_stages=collections.defaultdict(list)
                window_work=collections.defaultdict(collections.Counter)
                incomplete=[]
                ranks=collections.defaultdict(collections.Counter); windows=collections.defaultdict(collections.Counter)
                phase_rids=set(); cache=collections.Counter(); pairs=collections.Counter()
                starts=[r['metadata']['request_start_ns'] for r in cohort if r['metadata'].get('request_start_ns')]
                ends=[r['metadata']['request_end_ns'] for r in cohort if r['metadata'].get('request_end_ns')]
                beginning=min(starts) if starts else 0
                for record in cohort:
                    meta=record['metadata']; rid=canonical(meta.get('x_request_id')); phase_rids.add(rid)
                    bucket=int((meta.get('request_start_ns',beginning)-beginning)/60e9)
                    counts['client_records']+=1
                    if meta.get('was_cancelled'):counts['cancelled']+=1
                    if record.get('error'):counts['error_records']+=1
                    both={}; entry={'concurrency':c,'phase':phase,'rid':rid,'metadata':meta,'metrics':record.get('metrics',{})}
                    for role in ('prefill','decode'):
                        candidates=summaries.get((rid,role),[])
                        if not candidates:
                            counts[f'missing_{role}']+=1
                            history=request_events.get((rid,role),[])
                            last=max(history,key=lambda x:x['mono_ns']) if history else None
                            incomplete.append(dict(rid=rid,role=role,cancelled=meta.get('was_cancelled',False),
                                last_event=last,client_error=record.get('error')))
                            continue
                        if len(candidates)>1:counts[f'multiple_{role}']+=1
                        row=max(candidates,key=lambda x:x['wall_ns']); both[role]=row
                        rank=str(row['dp_rank']); ds=durations(row)
                        entry[role]=dict(row,durations_ms=ds)
                        counts[f'joined_{role}']+=1
                        ranks[role][rank]+=1
                        if role=='prefill':
                            miss=max(0,row['input_tokens']-row['cached_device']-row['cached_host']-row['cached_storage'])
                            for key in ('input_tokens','cached_device','cached_host','cached_storage'):
                                cache[key]+=row[key]
                            cache['miss_tokens']+=miss
                            ranks['prefill_miss_tokens'][rank]+=miss
                            ranks['prefill_host_tokens'][rank]+=row['cached_host']
                            work=window_work[f'{bucket}/{rank}']
                            work.update(input_tokens=row['input_tokens'],miss_tokens=miss,
                                host_tokens=row['cached_host'],requests=1)
                            stratum=f"{bin_tokens(row['input_tokens'])}/miss-{bin_tokens(miss)}/host-{bin_tokens(row['cached_host'])}"
                        else:
                            stratum=f"{bin_tokens(row['input_tokens'])}/output-{bin_tokens(row['output_tokens'])}"
                            ranks['decode_input_tokens'][rank]+=row['input_tokens']
                            ranks['decode_output_tokens'][rank]+=row['output_tokens']
                        windows[(role,bucket)][rank]+=1
                        for key,value in ds.items():
                            if value is not None:
                                stage[f'{role}/{key}'].append(value)
                                rank_stage[f'{role}/{rank}/{key}'].append(value)
                                stratified[f'{role}/{stratum}/{key}'].append(value)
                                window_stages[f'{bucket}/{role}/{key}'].append(value)
                    if len(both)==2:
                        counts['paired']+=1
                        if both['prefill']['room']!=both['decode']['room']:counts['room_mismatch']+=1
                        pairs[f"{both['prefill']['dp_rank']}->{both['decode']['dp_rank']}"]+=1
                    joined.write(json.dumps(entry,separators=(',',':'))+'\n')
                admission=collections.Counter(); blocked=collections.defaultdict(set)
                for event in events:
                    rid=canonical(event.get('rid'))
                    if rid in phase_rids and event.get('event')=='admission':
                        reason=event['reason'];admission[reason]+=1
                        if reason!='admitted':blocked[reason].add(rid)
                rank_totals={k:dict(v) for k,v in ranks.items()}
                imbalance={}
                for key,values in ranks.items():
                    vals=[values.get(str(rank),0) for rank in range(8)]
                    avg=statistics.fmean(vals)
                    imbalance[key]=dict(cv=statistics.pstdev(vals)/avg if avg else None,
                        max_min=max(vals)/min(vals) if min(vals)>0 else None,rank_totals=vals)
                phase_result=dict(coverage=dict(counts),start_ns=beginning,end_ns=max(ends) if ends else None,
                    stages_ms={k:stats(v) for k,v in stage.items()},
                    rank_stages_ms={k:stats(v) for k,v in rank_stage.items()},
                    stratified_stages_ms={k:stats(v) for k,v in stratified.items()},
                    rank_totals=rank_totals,rank_imbalance=imbalance,pd_pairs=dict(pairs),cache=dict(cache),
                    admission_event_counts=dict(admission),blocked_request_counts={k:len(v) for k,v in blocked.items()},
                    minute_stage_ms={k:stats(v) for k,v in window_stages.items()},
                    minute_prefill_work={k:dict(v) for k,v in window_work.items()},
                    incomplete_requests=incomplete,
                    minute_arrivals={f'{k[0]}/{k[1]}':dict(v) for k,v in windows.items()})
                if cache['input_tokens']:
                    phase_result['cache']['hit_rate']=1-cache['miss_tokens']/cache['input_tokens']
                point_result['phases'][phase]=phase_result
            result['points'][str(c)]=point_result
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# AUS request-level 阶段报告','',f"生成时间：{result['generated_at']}",'',
           '本表仅使用客户端 profiling cohort。阶段单位为 ms；取消/缺失不视为零。','']
    for c,point in result['points'].items():
        profile=point['phases'].get('profiling')
        if not profile:
            lines += [f'C{c}：尚无 profiling 原始记录。',''];continue
        lines += [f'## C{c}','',f"关联覆盖：`{json.dumps(profile['coverage'])}`",'',
                  '| 阶段 | 样本数 | mean | p50 | p90 | p99 |','|---|---:|---:|---:|---:|---:|']
        for stage,s in sorted(profile['stages_ms'].items()):
            lines.append(f"| {stage} | {s['n']} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p90']:.2f} | {s['p99']:.2f} |")
        lines += ['',f"发生 allocation 阻塞的请求：`{json.dumps(profile['blocked_request_counts'])}`",'',
                  f"缓存统计：`{json.dumps(profile['cache'])}`",'',
                  '逐rank负载、60秒到达分布、长度/cache分层与P/D配对矩阵见 summary.json。','']
    lines+=['## 解释边界','']+['- '+x for x in result['limitations']]
    (out/'REPORT.zh-CN.md').write_text('\n'.join(lines)+'\n')
    print(out/'summary.json')

if __name__=='__main__':main()
