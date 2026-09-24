#!/usr/bin/env python3
"""Audit shadow decisions and bounded probes, without computing performance gains."""
import collections,datetime,json,math,re
from pathlib import Path
root=Path('/perf_apps/liyingli/bench_agentx/router-r234-31699-20260924');run=root/'runs/r234-31699-control';out=root/'observation';out.mkdir(exist_ok=True)
groups=collections.defaultdict(list);picks={}
def number(v):
 if v is None or v=='None':return None
 return float(v[5:-1] if v.startswith('Some(') else v)
def timestamp(s):return int(datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()*1e9)
for raw in open(run/'server-logs/router.log'):
 s=re.sub(r'\x1b\[[0-9;]*m','',raw)
 def field(k):
  m=re.search(r'(?<!\w)'+re.escape(k)+r'=(\S+)',s);return m[1] if m else None
 if 'routing experiment candidate' in s:
  keys=['decision_id','role','target','selected','legacy_selected','demand_suggested','tier_suggested','demand_known','input_tokens','work_tokens','demand_cost','legacy_cost','tier_cost','gpu_hits','host_hits']
  row={k:field(k) for k in keys};row['time_ns']=timestamp(s.split()[1]);row['decision_id']=int(row['decision_id'])
  for k in ['input_tokens','work_tokens','demand_cost','legacy_cost','tier_cost','gpu_hits','host_hits']:row[k]=number(row[k])
  row['rank']=int(row['target'].split('#dp')[-1])
  row['stats']={k:int(re.search(k+r': (\d+)',s)[1]) for k in ['device_stores','host_stores','unknown_events','rejected_stores']}
  groups[(row['role'],row['decision_id'])].append(row)
 elif 'pick policy=' in s and 'kv-aware' in s and field('decision_id'):
  picks[int(field('decision_id'))]={k:number(field(k)) for k in ['cache_hits','w_overlap']}
checks={};summary={};examples={};flat=[]
for role,flag,cost in [('Decode','demand_suggested','demand_cost'),('Prefill','tier_suggested','tier_cost')]:
 count=changes=bad_selected=bad_suggested=bad_min=bad_shape=unknown=bad_work=0;example=[]
 for (r,decision),rows in groups.items():
  if r!=role:continue
  count+=1;selected=[v for v in rows if v['selected']=='true'];legacy=[v for v in rows if v['legacy_selected']=='true'];suggested=[v for v in rows if v[flag]=='true']
  bad_shape+=len(rows)!=8 or len({v['target'] for v in rows})!=8
  if len(selected)!=1 or len(legacy)!=1 or selected[0]['target']!=legacy[0]['target']:bad_selected+=1
  if len(suggested)!=1:bad_suggested+=1;continue
  if abs(suggested[0][cost]-min(v[cost] for v in rows))>1e-6:bad_min+=1
  if selected[0]['target']!=suggested[0]['target']:
   changes+=1
   if len(example)<3:example.append({'decision_id':decision,'old':selected[0],'suggested':suggested[0]})
  if role=='Decode':
   unknown+=any(v['demand_known']!='true' for v in rows)
   bad_work+=any(v['input_tokens'] is None or v['work_tokens']!=v['input_tokens'] or v['demand_cost']<v['work_tokens'] for v in rows)
  flat.append({'role':role,'decision_id':decision,'selected':selected[0]['rank'],'suggested':suggested[0]['rank'],'input_tokens':selected[0]['input_tokens'],'time_ns':selected[0]['time_ns']})
 summary[role]={'decisions':count,'different_suggestions':changes,'different_pct':changes/count*100,'actual_vs_legacy_mismatch':bad_selected,'missing_suggestion':bad_suggested,'nonminimum_suggestion':bad_min,'bad_candidate_sets':bad_shape,'unknown_demand_decisions':unknown,'bad_input_work':bad_work}
 checks[role+'_shadow_and_argmin']=not any([bad_selected,bad_suggested,bad_min,bad_shape,unknown,bad_work]);examples[role]=example
engine={}
for p in (run/'diagnostics').glob('*/*.jsonl'):
 for line in open(p):
  d=json.loads(line)
  if d.get('event')=='request_summary':engine[(d['role'],d['rid'])]=d
engine_requests=[d for (role,rid),d in engine.items() if role=='decode' and (re.fullmatch(r'[0-9a-f-]{36}',rid) or rid.startswith(('aus-smoke-','guard-stream-smoke','obs-')))]
a=collections.Counter((x['selected'],int(x['input_tokens'])) for x in flat if x['role']=='Decode')
b=collections.Counter((x['dp_rank'],x['input_tokens']) for x in engine_requests)
checks['all_decode_rank_length_multiset_matches_engine']=a==b
summary['rank_length_reconciliation']={'router_decisions':sum(a.values()),'engine_requests':sum(b.values()),'router_only':[[list(k),v] for k,v in (a-b).items()][:15],'engine_only':[[list(k),v] for k,v in (b-a).items()][:15],'scope':'Aggregate rank+length reconciliation; request-ID checks below use controlled probes.'}
probes=json.loads((out/'probes.json').read_text());probe_groups={};probe_checks=[]
for probe in probes:
 expected=probe['usage']['prompt_tokens'];rid=probe['rid']
 matched=[rows for (role,_),rows in groups.items() if role=='Decode' and rows[0]['input_tokens']==expected and probe['start_ns']-5_000_000<=rows[0]['time_ns']<=probe['end_ns']+5_000_000]
 ok=len(matched)==1
 if ok:
  g=matched[0];chosen=next(v for v in g if v['selected']=='true');d=engine.get(('decode',rid),{})
  ok=d.get('input_tokens')==expected and d.get('dp_rank')==chosen['rank'];probe_groups[rid]=g
 probe_checks.append({'rid':rid,'matched_group_count':len(matched),'rank_and_length_match':ok})
checks['controlled_probe_rank_and_length']=all(x['rank_and_length_match'] for x in probe_checks)
empty={rid:all(abs(x['demand_cost']-x['work_tokens'])<1e-6 for x in probe_groups.get(rid,[])) and bool(probe_groups.get(rid)) for rid in ['obs-empty-before','obs-empty-after']}
checks['before_and_after_ledger_empty']=all(empty.values())
concurrent=[p for p in probes if p['rid'].startswith('obs-demand-')]
ordered=sorted(concurrent,key=lambda p:sum(x['demand_cost']-x['work_tokens'] for x in probe_groups[p['rid']]))
bookings=collections.defaultdict(float);ledger_steps=[]
for probe in ordered:
 g=probe_groups[probe['rid']];chosen=next(x for x in g if x['selected']=='true')
 observed={x['rank']:x['demand_cost']-x['work_tokens'] for x in g}
 ok=all(abs(observed[k]-bookings[k])<1e-6 for k in observed)
 ledger_steps.append({'rid':probe['rid'],'rank':chosen['rank'],'input_tokens':chosen['input_tokens'],'prior_load_by_rank':observed,'matches_sum_of_prior_bookings':ok})
 bookings[chosen['rank']]+=chosen['input_tokens']
last_decision=max(x['time_ns'] for p in concurrent for x in probe_groups[p['rid']]);first_end=min(p['end_ns'] for p in concurrent)
checks['controlled_concurrent_ledger_additive']=all(s['matches_sum_of_prior_bookings'] for s in ledger_steps) and first_end-last_decision>50_000_000
repeat=next(p for p in probes if p['rid']=='obs-cache-repeat');pg=[rows for (role,_),rows in groups.items() if role=='Prefill' and repeat['start_ns']-5_000_000<=rows[0]['time_ns']<=repeat['end_ns']+5_000_000]
repeat_result={'matched_groups':len(pg)}
if len(pg)==1:
 chosen=next(x for x in pg[0] if x['selected']=='true');d=engine[('prefill',repeat['rid'])];repeat_result.update({'router_rank':chosen['rank'],'engine_rank':d['dp_rank'],'router_gpu_tokens':chosen['gpu_hits']*64,'router_host_tokens':chosen['host_hits']*64,'engine_cached_device':d['cached_device'],'engine_cached_host':d['cached_host']})
 checks['controlled_gpu_cache_matches_engine']=chosen['rank']==d['dp_rank'] and chosen['gpu_hits']*64==d['cached_device'] and d['cached_device']>0
else:checks['controlled_gpu_cache_matches_engine']=False
p_rows=[v for (role,_),rows in groups.items() if role=='Prefill' for v in rows];host_rows=[v for v in p_rows if v['host_hits']>0]
score_errors=0
for (role,i),rows in groups.items():
 if role!='Prefill':continue
 chosen=next(v for v in rows if v['selected']=='true');pick=picks[i]
 expected=chosen['legacy_cost']+pick['w_overlap']*(pick['cache_hits']-chosen['gpu_hits']-.5*chosen['host_hits'])
 score_errors+=abs(expected-chosen['tier_cost'])>1e-6
checks['tier_scoring_formula']=score_errors==0
checks['live_tier_events_recognized']=max(v['stats']['host_stores'] for v in p_rows)>0 and not any(v['stats']['unknown_events'] or v['stats']['rejected_stores'] for v in p_rows)
result={'scope':'8K shadow correctness only; not performance evidence','passed':all(checks.values()),'checks':checks,'decisions':summary,'examples':examples,'controlled_probes':probe_checks,'empty_ledger_probes':empty,'concurrent_ledger':ledger_steps,'concurrent_completion_margin_ms':(first_end-last_decision)/1e6,'cache_repeat':repeat_result,'host_only_candidates':host_rows,'host_observations':{'candidate_rows_with_host_hits':len(host_rows),'max_host_hit_tokens':max([v['host_hits']*64 for v in host_rows] or [0]),'engine_requests_with_host_reuse':sum(d.get('cached_host',0)>0 for (role,_),d in engine.items() if role=='prefill')},'limits':['Actual routing stayed legacy; no on-mode throughput or latency benefit is inferred.','Host-only candidates were not selected; actual host-reuse execution is not established by this shadow run.']}
(out/'correctness.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'decisions.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in flat))
print(json.dumps({k:result[k] for k in ['passed','checks','decisions','empty_ledger_probes','concurrent_ledger','concurrent_completion_margin_ms','cache_repeat','host_observations']},indent=2))
