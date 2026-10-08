#!/usr/bin/env python3
"""Compare identical trace turns; report selection and workload changes explicitly."""
import argparse,collections,json,statistics
from pathlib import Path
KEYS=('conversation_id','turn_index','source_trace_id','source_outer_idx','source_inner_idx','source_kind')
def load(run):
 rows={};duplicates=set();total=0
 for line in (run/'analysis/joined-requests.jsonl').open():
  x=json.loads(line)
  if x['phase']!='profiling':continue
  total+=1;m=x['metadata'];key=tuple(m.get(k) for k in KEYS)
  if not key[0] or key[1] is None or not key[2]:raise ValueError('Missing stable trace identity')
  if key in rows:duplicates.add(key)
  rows[key]=x
 for key in duplicates:del rows[key]
 return rows,{'profile_records':total,'unique_keys':len(rows),'ambiguous_keys_excluded':len(duplicates)}
def values(x):
 p=x['prefill'];d=x['decode'];m=x['metrics']
 return {'input_tokens':p['input_tokens'],'output_tokens':m['output_sequence_length']['value'],
  'device_tokens':p['cached_device'],'host_tokens':p['cached_host'],
  'miss_tokens':p['input_tokens']-sum(p.get(k,0) for k in ('cached_device','cached_host','cached_storage')),
  'ttft_ms':m['time_to_first_token']['value'],
  **{'prefill_'+k:v for k,v in p['durations_ms'].items()},
  **{'decode_'+k:v for k,v in d['durations_ms'].items()}}
def compare(a,b,input_tolerance=0):
 left,la=load(a);right,ra=load(b);common=left.keys()&right.keys();pairs=[];excluded=collections.Counter()
 for key in sorted(common,key=str):
  x,y=left[key],right[key]
  if not all(z.get('prefill') and z.get('decode') for z in (x,y)):excluded['unpaired_backend']+=1;continue
  u,v=values(x),values(y)
  if abs(u['input_tokens']-v['input_tokens'])>input_tolerance or abs(u['input_tokens']-v['input_tokens'])>0.001*max(u['input_tokens'],v['input_tokens']):excluded['input_length_mismatch']+=1;continue
  if u['output_tokens']!=v['output_tokens']:excluded['actual_output_length_mismatch']+=1;continue
  pairs.append((key,u,v))
 metrics={}
 if pairs:
  for field in pairs[0][1]:
   u=[x[field] for _,x,y in pairs];v=[y[field] for _,x,y in pairs];d=[y-x for x,y in zip(u,v)]
   metrics[field]={'n':len(u),'reference_mean':statistics.mean(u),'candidate_mean':statistics.mean(v),'mean_change_pct':(sum(v)/sum(u)-1)*100 if sum(u) else None,'paired_delta_mean':statistics.mean(d),'paired_delta_p50':statistics.median(d)}
 pertrace=collections.defaultdict(lambda:collections.Counter())
 for key,u,v in pairs:
  t=pertrace[key[2]];t['n']+=1
  for field in ('miss_tokens','ttft_ms','prefill_forward_envelope_ms','prefill_queue_ms'):
   t['reference_'+field]+=u[field];t['candidate_'+field]+=v[field]
 return {'reference':str(a),'candidate':str(b),'identity_fields':KEYS,'input_token_tolerance':input_tolerance,'input_relative_tolerance':0.001,'reference_accounting':la,'candidate_accounting':ra,'common_identity_keys':len(common),'reference_only_keys':len(left.keys()-right.keys()),'candidate_only_keys':len(right.keys()-left.keys()),'exclusions':dict(excluded),'matched_pairs':len(pairs),'metrics':metrics,'per_source_trace':dict(pertrace),'limitations':['Intersection of completed profiling trace turns is selected by both runs; it is not an unbiased throughput estimator.','Equal input/output lengths do not prove byte-identical prompts; stable trace revision and harness provenance must be checked separately.','Turns sharing traces, cache state and time are dependent; request count is not an independent replicate count.','Sequential single-run differences need rollback/repetition before claiming stable benefit.','Forward envelopes include scheduling gaps, not exclusive GPU compute.']}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('reference',type=Path);p.add_argument('candidate',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--input-tolerance',type=int,choices=range(0,9),default=0);a=p.parse_args()
 result=compare(a.reference,a.candidate,a.input_tolerance);a.output.mkdir(parents=True,exist_ok=True);(a.output/'matched-requests.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# 相同 trace turn 配对比较','',f"共同身份 {result['common_identity_keys']}；输入容差≤{a.input_tolerance} tokens 且≤0.1%、实际输出长度相同、两端关联的配对 {result['matched_pairs']}。",'', '| 指标 | 对照 mean | 处理 mean | 变化 |','|---|---:|---:|---:|']
 for k,x in result['metrics'].items():
  change=f"{x['mean_change_pct']:+.2f}%" if x['mean_change_pct'] is not None else 'n/a'
  lines.append(f"| {k} | {x['reference_mean']:.4f} | {x['candidate_mean']:.4f} | {change} |")
 lines+=['','这是已完成请求交集，存在选择效应；不能用交集推导总体吞吐收益，也不能把请求数当作独立实验重复数。原始配对覆盖、排除原因、每条 source trace 的统计见 JSON。','']
 (a.output/'MATCHED-REQUESTS.zh-CN.md').write_text('\n'.join(lines));print(json.dumps({k:result[k] for k in ('common_identity_keys','matched_pairs','exclusions')}))
if __name__=='__main__':main()
