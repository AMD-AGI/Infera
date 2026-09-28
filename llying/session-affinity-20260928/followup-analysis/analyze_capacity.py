"""Read-only audit of P imbalance, selected chunk traces and D memory samples."""
import collections,datetime,json,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent
RUN=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-performance')
start=datetime.datetime.fromisoformat('2026-09-28T07:17:40.481+00:00').timestamp()
end=start+3600

def stat(v):
 v=sorted(v)
 return dict(n=len(v),mean=statistics.fmean(v),p50=statistics.median(v),p95=v[int(.95*(len(v)-1))],max=max(v)) if v else dict(n=0)
qrows=[];kvrows=[];dmem=collections.defaultdict(list)
for line in (RUN/'sampling/engine.jsonl').open():
 x=json.loads(line)
 if not x.get('series'):continue
 t=datetime.datetime.fromisoformat(x['captured_at']).timestamp()
 if not start<=t<end:continue
 vals=collections.defaultdict(dict)
 for s in x['series']:
  rank=s.get('labels',{}).get('dp_rank')
  if rank is not None:vals[s['metric']][int(rank)]=s['value']
 if x.get('endpoint')=='prefill':
  qs=vals['sglang:num_queue_reqs']
  if len(qs)==8:qrows.append([qs[i] for i in range(8)])
  used=vals['sglang:kv_used_tokens'];evict=vals['sglang:kv_evictable_tokens'];avail=vals['sglang:kv_available_tokens'];total=vals['sglang:max_total_num_tokens']
  if all(len(a)==8 for a in [used,evict,avail,total]):kvrows.append([(used[i],evict[i],avail[i],total[i]) for i in range(8)])
 else:
  for k in ['sglang:kv_cache_memory_usage_gb','sglang:graph_memory_usage_gb','sglang:max_total_num_tokens']:
   if vals[k]:dmem[k].extend(vals[k].values())
nonzero=[v for v in qrows if sum(v)>0]
summary=dict(samples=len(qrows),queue_per_rank=[stat([v[i] for v in qrows]) for i in range(8)],queue_total=stat([sum(v) for v in qrows]),nonzero_queue_samples=len(nonzero),queue_cv_nonzero=stat([statistics.pstdev(v)/statistics.fmean(v) for v in nonzero]),queue_nonzero_and_other_rank_zero=sum(min(v)==0 for v in nonzero),max_queue_ge4_other_zero=sum(max(v)>=4 and min(v)==0 for v in qrows),max_queue_ge8_other_zero=sum(max(v)>=8 and min(v)==0 for v in qrows),d_memory={k:stat(v) for k,v in dmem.items()},limitations=['Empty local queue is not proof of an idle independent GPU: DP attention ranks share TP computation.','Two-second samples do not contain decision-time complete cache state.'])
rows=[]
for line in (RUN/'analysis/joined-requests.jsonl').open():
 r=json.loads(line)
 if r.get('phase')=='profiling' and r.get('prefill'):
  p=r['prefill'];m=r['metadata'];rows.append(dict(rank=p['dp_rank'],miss=p['input_tokens']-sum(p.get(k,0) for k in ['cached_device','cached_host','cached_storage']),input=p['input_tokens'],turn=m['turn_index'],session=m['x_correlation_id']))
summary['cumulative']={str(i):dict(requests=sum(r['rank']==i for r in rows),miss=sum(r['miss'] for r in rows if r['rank']==i),miss_gt128k=sum(r['rank']==i and r['miss']>131072 for r in rows),miss_gt128k_turn0=sum(r['rank']==i and r['miss']>131072 and r['turn']==0 for r in rows)) for i in range(8)}
(OUT/'p-balance-and-d-memory.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
# Parent roots carry rid and role; keep chunk spans per trace without loading all spans.
window=json.loads((OUT/'tail-blocking-windows.json').read_text());targets=set();seen=set()
for m in window['matches']:
 b=m['overlapping_long_prefill']
 if b and b['rid'] not in seen:
  seen.add(b['rid']);targets.update([b['rid'],m['tail']['rid']])
 if len(seen)==3:break
roots={};chunks=collections.defaultdict(list);selected=collections.defaultdict(list);dedup=set()
for line in (RUN/'traces/spans.jsonl').open():
 x=json.loads(line)
 if x.get('record_type')!='span':continue
 key=(x['trace_id'],x['span_id'])
 if key in dedup:continue
 dedup.add(key)
 attrs=x.get('attributes',{});rid=str(attrs.get('rid','')).split('_')[-1]
 if rid in targets and x['name'].lower().startswith('prefill '):roots[x['trace_id']]=dict(rid=rid,attrs=attrs,name=x['name'])
 if x['name']=='chunked_prefill':chunks[x['trace_id']].append(dict(start_ns=x['start_time_ns'],end_ns=x['end_time_ns']))
 if x['name'] in ['prefill_forward','prefill_waiting']:selected[x['trace_id']].append(dict(name=x['name'],start_ns=x['start_time_ns'],end_ns=x['end_time_ns']))
report={}
for trace,root in roots.items():
 c=sorted(chunks[trace],key=lambda x:x['start_ns']);ds=[(x['end_ns']-x['start_ns'])/1e6 for x in c]
 report[root['rid']]=dict(**root,trace_id=trace,chunk_count=len(c),chunk_envelope_ms=stat(ds),sum_chunk_ms=sum(ds),chunks=c,stages=selected[trace])
(OUT/'selected-chunk-traces.json').write_text(json.dumps(report,indent=2)+'\n')
print('chunks',{k:{x:v[x] for x in ['chunk_count','sum_chunk_ms','chunk_envelope_ms']} for k,v in report.items()})
