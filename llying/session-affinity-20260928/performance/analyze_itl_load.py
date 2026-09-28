"""Associate matched requests with sampled Decode load during generation; no causal claim."""
import bisect,collections,datetime,json,math,statistics
from pathlib import Path
BASE=Path('/perf_apps/liyingli/bench_agentx/p8d8-adaptive-31625-20260923/runs/g0-guard-completion')
RUN=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-performance')
KEYS=('conversation_id','turn_index','source_trace_id','source_outer_idx','source_inner_idx','source_kind')
def stats(xs):
 xs=sorted(x for x in xs if x is not None and math.isfinite(x))
 if not xs:return {'n':0}
 return {'n':len(xs),'mean':statistics.fmean(xs),'p50':statistics.median(xs),'p90':xs[int(.9*(len(xs)-1))]}
def load(run):
 samples=[]
 for line in (run/'sampling/engine.jsonl').open():
  d=json.loads(line)
  if d.get('endpoint')!='decode' or 'series' not in d:continue
  metrics=collections.defaultdict(dict)
  for s in d['series']:
   if s['metric'] not in ['sglang:num_running_reqs','sglang:kv_used_tokens','sglang:num_decode_transfer_queue_reqs']:continue
   rank=s['labels'].get('dp_rank')
   if rank is not None:metrics[s['metric']][int(rank)]=s['value']
  bs=metrics['sglang:num_running_reqs'];kv=metrics['sglang:kv_used_tokens'];wait=metrics['sglang:num_decode_transfer_queue_reqs']
  if len(bs)!=8 or len(kv)!=8:continue
  t=datetime.datetime.fromisoformat(d['captured_at']).timestamp()
  samples.append((t,[bs[i] for i in range(8)]+[sum(bs.values()),max(bs.values()),sum(kv.values()),sum(wait.values())]))
 samples.sort();ts=[t for t,_ in samples];prefix=[[0.] for _ in range(12)]
 for _,vs in samples:
  for j,v in enumerate(vs):prefix[j].append(prefix[j][-1]+v)
 rows={};duplicate=set()
 for line in (run/'analysis/joined-requests.jsonl').open():
  r=json.loads(line)
  if r['phase']!='profiling' or not r.get('prefill') or not r.get('decode'):continue
  key=tuple(r['metadata'].get(k) for k in KEYS)
  if key in rows:duplicate.add(key)
  d=r['decode'];m=r['metrics'];n=m.get('output_sequence_length',{}).get('value',0);itl=m.get('inter_token_latency',{}).get('value')
  t=d['times'];offset=d['wall_ns']/1e9-d['mono_ns']/1e9;start=offset+t['forward_entry_time'];end=offset+t['completion_time']
  a,b=bisect.bisect_left(ts,start),bisect.bisect_right(ts,end);count=b-a
  x={'input':d['input_tokens'],'output':n,'itl_ms':itl,'gen_ms':d['durations_ms']['generation_ms'],'rank':d['dp_rank'],'samples':count}
  if count>=3:
   avg=[(v[b]-v[a])/count for v in prefix]
   x.update(own_batch=avg[d['dp_rank']],total_batch=avg[8],max_rank_batch=avg[9],kv_tokens=avg[10],waiting_kv=avg[11])
  rows[key]=x
 for k in duplicate:rows.pop(k,None)
 return rows
left=load(BASE);right=load(RUN);pairs=[]
for key in left.keys()&right.keys():
 x,y=left[key],right[key]
 if abs(x['input']-y['input'])>8 or abs(x['input']-y['input'])>.001*max(x['input'],y['input']) or x['output']!=y['output']:continue
 if x['itl_ms'] is None or y['itl_ms'] is None:continue
 pairs.append((x,y))
def summary(ps):
 return {k:{'reference':stats([x.get(k) for x,y in ps]),'candidate':stats([y.get(k) for x,y in ps])} for k in ['input','output','itl_ms','gen_ms','own_batch','total_batch','max_rank_batch','kv_tokens']}
covered=[(x,y) for x,y in pairs if 'total_batch' in x and 'total_batch' in y]
comparable=[(x,y) for x,y in covered if abs(x['total_batch']-y['total_batch'])<=2 and abs(x['own_batch']-y['own_batch'])<=.5]
comparable_kv=[(x,y) for x,y in comparable if abs(x['kv_tokens']-y['kv_tokens'])<=.1*max(x['kv_tokens'],y['kv_tokens'])]
more=[(x,y) for x,y in covered if y['total_batch']-x['total_batch']>=8]
less=[(x,y) for x,y in covered if y['total_batch']-x['total_batch']<=-8]
bins={}
for name,rows in [('G0',left),('P_session',right)]:
 groups=collections.defaultdict(list)
 for x in rows.values():
  if 'total_batch' not in x or x['itl_ms'] is None or not(32768<=x['input']<131072 and 512<=x['output']<=2048):continue
  label=next((f'<= {cut}' for cut in [16,32,48,64] if x['total_batch']<=cut),'> 64')
  groups[label].append(x)
 bins[name]={k:{'itl_ms':stats([x['itl_ms'] for x in v]),'input':stats([x['input'] for x in v]),'total_batch':stats([x['total_batch'] for x in v])} for k,v in groups.items()}
result={'all_matched_with_itl':summary(pairs),'matched_with_at_least_3_samples_each':summary(covered),'similar_batch_total_within2_own_within0_5':summary(comparable),'similar_batch_and_kv_within10pct':summary(comparable_kv),'candidate_total_batch_at_least8_larger':summary(more),'candidate_total_batch_at_least8_smaller':summary(less),'within_run_bins_input32k_to128k_output512_to2048':bins,'limitations':['Samples are about 2 seconds apart; num_running_reqs is not exact per-step batch size.','Load during generation is affected by service duration, so associations do not establish causal direction.','KV used includes requests waiting for KV, not only generating sequences.','Similar-batch filtering is a selected subset; different Decode nodes remain a confound.']}
(RUN/'analysis/itl-load-analysis.json').write_text(json.dumps(result,indent=2)+'\n')
for name,v in result.items():
 if name in ['limitations','within_run_bins_input32k_to128k_output512_to2048']:continue
 print(name, {k:{side:round(z['mean'],3) if z.get('n') else None for side,z in values.items()} for k,values in v.items() if k in ['itl_ms','gen_ms','own_batch','total_batch']},'n',v['itl_ms']['candidate']['n'])
print('bins',json.dumps(bins))
