#!/usr/bin/env python3
"""Compare completed warmup cohorts without treating interrupted profiling as a result."""
import argparse,collections,importlib.util,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',type=Path,nargs=3);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=Path(__file__).resolve().parents[2]/'p8d8-c80-c112-tracing-aus-20260922/scripts/analyze.py'
spec=importlib.util.spec_from_file_location('base_analysis',source);base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
keys=('conversation_id','turn_index','source_trace_id','source_outer_idx','source_inner_idx','source_kind')
def load(r):
 events={}
 for f in (r/'diagnostics').glob('*/*.jsonl'):
  for x in base.jsonl(f):
   if x.get('event')=='request_summary':events[(base.canonical(x['rid']),x['role'])]=x
 raw=[x for x in base.jsonl(r/'c80/aiperf_artifacts/profile_export.jsonl') if x.get('metadata',{}).get('benchmark_phase')=='warmup'];raw.sort(key=lambda x:x['metadata']['request_start_ns']);rows={};counts=collections.Counter();overall=collections.defaultdict(list)
 for i,x in enumerate(raw):
  m=x['metadata'];key=tuple(m.get(k) for k in keys);assert key not in rows
  counts['records']+=1;counts['errors']+=bool(x.get('error'));rid=base.canonical(m['x_request_id']);ps=events.get((rid,'prefill'));ds=events.get((rid,'decode'));counts['paired']+=bool(ps and ds)
  if x.get('error') or not ps or not ds:continue
  met=x['metrics'];v={'input_tokens':ps['input_tokens'],'output_tokens':met['output_sequence_length']['value'],'ttft_ms':met['time_to_first_token']['value'],'latency_ms':met['request_latency']['value'],'miss_tokens':ps['input_tokens']-sum(ps.get(k,0) for k in ['cached_device','cached_host','cached_storage']),**{'prefill_'+k:z for k,z in base.durations(ps).items()},**{'decode_'+k:z for k,z in base.durations(ds).items()}}
  rows[key]={'group':'first84_primers' if i<84 else 'remaining800','values':v}
  for k,z in v.items():overall[k].append(z)
 return rows,{'accounting':dict(counts),'output_lengths':dict(collections.Counter(x['values']['output_tokens'] for x in rows.values())),'stages':{k:base.stats(v) for k,v in overall.items()},'primer_start_span_s':(raw[83]['metadata']['request_start_ns']-raw[0]['metadata']['request_start_ns'])/1e9,'next_group_start_offset_s':(raw[84]['metadata']['request_start_ns']-raw[0]['metadata']['request_start_ns'])/1e9}
loaded=[load(r) for r in a.runs];report={'cases':{r.name:x[1] for r,x in zip(a.runs,loaded)},'comparisons':{},'limitations':['All valid warmup outputs have one token; they do not reproduce profiling decode residency or P guard over-hold.','Warmup timing includes cold-prefix work, startup ramp and DAG/idle dependencies; it is not profiling throughput.','Completed A1 warmup is usable as auxiliary evidence although its profiling was preempted and invalid.','Matched completed turns are a selected cohort; identical length does not guarantee identical bytes.']}
for i,j in [(0,1),(0,2),(1,2)]:
 left,right=loaded[i][0],loaded[j][0];common=left.keys()&right.keys();groups=collections.defaultdict(list);excluded=collections.Counter()
 for key in common:
  u,v=left[key]['values'],right[key]['values']
  if abs(u['input_tokens']-v['input_tokens'])>8 or abs(u['input_tokens']-v['input_tokens'])>.001*max(u['input_tokens'],v['input_tokens']):excluded['input_mismatch']+=1;continue
  if u['output_tokens']!=v['output_tokens']:excluded['output_mismatch']+=1;continue
  if left[key]['group']!=right[key]['group']:excluded['group_mismatch']+=1;continue
  groups['all'].append((u,v));groups[left[key]['group']].append((u,v))
 summary={}
 for group,pairs in groups.items():
  metrics={}
  for k in pairs[0][0]:
   valid=[(u[k],v[k]) for u,v in pairs if u[k] is not None and v[k] is not None];u=[x for x,y in valid];v=[y for x,y in valid]
   if not u:continue
   metrics[k]={'n':len(u),'reference_mean':statistics.mean(u),'candidate_mean':statistics.mean(v),'change_pct':(sum(v)/sum(u)-1)*100 if sum(u) else None}
  summary[group]=metrics
 report['comparisons'][a.runs[j].name+'_vs_'+a.runs[i].name]={'common_keys':len(common),'exclusions':dict(excluded),'groups':summary}
common3=set.intersection(*(set(x[0]) for x in loaded));triples=collections.defaultdict(list)
for key in common3:
 rows=[x[0][key] for x in loaded];vals=[x['values'] for x in rows];inputs=[x['input_tokens'] for x in vals]
 if max(inputs)-min(inputs)>8 or max(inputs)-min(inputs)>.001*max(inputs):continue
 if len({x['output_tokens'] for x in vals})!=1 or len({x['group'] for x in rows})!=1:continue
 triples['all'].append(vals);triples[rows[0]['group']].append(vals)
three={}
for group,rows in triples.items():
 three[group]={'n':len(rows),'cases':{}}
 for i,r in enumerate(a.runs):
  three[group]['cases'][r.name]={k:statistics.mean(z[i][k] for z in rows if z[i][k] is not None) for k in rows[0][i] if any(z[i][k] is not None for z in rows)}
report['three_way_same_cohort']=three
a.output.mkdir(parents=True,exist_ok=True);(a.output/'warmup-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
for name,c in report['comparisons'].items():
 print(name)
 for group,m in c['groups'].items():print(group,{k:round(m[k]['change_pct'],2) if m[k]['change_pct'] is not None else None for k in ['ttft_ms','prefill_queue_ms','prefill_forward_envelope_ms','miss_tokens']},'n=',m['ttft_ms']['n'])
print('accounting',{k:v['accounting'] for k,v in report['cases'].items()})
