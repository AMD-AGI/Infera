"""Reproduce routing figures and audit observed P tails; no counterfactual cache claim."""
import collections,csv,importlib.util,json,statistics,sys
from pathlib import Path
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[1]
RUN=Path('/perf_apps/liyingli/bench_agentx/session-affinity-31999-20260928/runs/session-affinity-31999-performance')

def write_csv(name,rows):
 if not rows:return
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit():
 data=[json.loads(l) for l in (RUN/'analysis/joined-requests.jsonl').open()]
 data=[r for r in data if r.get('prefill') and r.get('decode')]
 data.sort(key=lambda r:r['metadata']['request_start_ns'])
 prior=collections.defaultdict(list); rows=[]
 for r in data:
  m=r['metadata'];p=r['prefill'];t=p['times'];conv=m['conversation_id'];rank=p['dp_rank']
  prev=[x for x in prior[conv] if x['metadata']['turn_index']==m['turn_index']-1 and x['metadata']['request_end_ns']<=m['request_start_ns']]
  prev=prev[0] if len(prev)==1 else None
  hit=sum(p.get(k,0) for k in ('cached_device','cached_host','cached_storage'))
  row=dict(rid=r['rid'],binding_session=m['x_correlation_id'],root_session=m.get('root_correlation_id'),conversation=conv,turn=m['turn_index'],source_kind=m.get('source_kind'),p_rank=rank,input_tokens=p['input_tokens'],gpu_hit=p['cached_device'],host_hit=p['cached_host'],storage_hit=p.get('cached_storage',0),total_hit=hit,miss=p['input_tokens']-hit,p_queue_ms=p['durations_ms']['queue_ms'],p_bootstrap_ms=p['durations_ms']['bootstrap_ms'],forward_envelope_ms=p['durations_ms']['forward_envelope_ms'],transfer_tail_ms=p['durations_ms']['transfer_tail_ms'],ttft_ms=r['metrics'].get('time_to_first_token',{}).get('value'),previous_rid=prev['rid'] if prev else None,previous_p_rank=prev['prefill']['dp_rank'] if prev else None,previous_input_tokens=prev['prefill']['input_tokens'] if prev else None,append_only_prior_input_proxy=min(prev['prefill']['input_tokens'],p['input_tokens']) if prev else None,exact_theoretical_prefix=None,counterfactual_best_rank=None)
  # Previous input is only a conditional proxy: actual token LCP and residency are absent.
  if r['phase']=='profiling':rows.append(row)
  prior[conv].append(r)
 tails=sorted([r for r in rows if r['p_queue_ms']>10000],key=lambda r:-r['p_queue_ms'])
 write_csv('all-p-requests.csv',rows);write_csv('p-queue-over10s.csv',tails)
 def agg(rs):
  return dict(requests=len(rs),queue_seconds=sum(r['p_queue_ms'] for r in rs)/1000,mean_queue_ms=statistics.fmean(r['p_queue_ms'] for r in rs) if rs else None,mean_miss=statistics.fmean(r['miss'] for r in rs) if rs else None,host_requests=sum(r['host_hit']>0 for r in rs),host_tokens=sum(r['host_hit'] for r in rs),miss_tokens=sum(r['miss'] for r in rs),input_tokens=sum(r['input_tokens'] for r in rs))
 ranks={str(k):dict(all=agg([r for r in rows if r['p_rank']==k]),tails=agg([r for r in tails if r['p_rank']==k])) for k in range(8)}
 groups=collections.defaultdict(list)
 for r in tails:groups[r['binding_session']].append(r)
 sessions=sorted([dict(binding_session=k,conversation=rs[0]['conversation'],root_session=rs[0]['root_session'],ranks=sorted(set(r['p_rank'] for r in rs)),**agg(rs)) for k,rs in groups.items()],key=lambda r:-r['queue_seconds'])
 buckets={}
 for name,fn in [('miss_le4k',lambda r:r['miss']<=4096),('miss_gt32k',lambda r:r['miss']>32768),('host_positive',lambda r:r['host_hit']>0),('no_host',lambda r:r['host_hit']==0),('first_turn',lambda r:r['turn']==0),('later_turn',lambda r:r['turn']>0),('hit_ge95pct',lambda r:r['total_hit']>=.95*r['input_tokens'])]:buckets[name]=agg([r for r in tails if fn(r)])
 result=dict(all=agg(rows),tails=agg(tails),tail_binding_sessions=len(groups),tail_root_sessions=len(set(r['root_session'] for r in tails)),by_rank=ranks,tail_buckets=buckets,top_sessions=sessions[:20],top_requests=tails[:15],limits=['Exact token-prefix LCP and candidate residency are not present in joined request records.','append_only_prior_input_proxy assumes prior input is unchanged and retained; not a guaranteed hit. It excludes previous Decode output.','Host tokens measure observed cache accounting, not isolated host copy duration.','No complete decision-time candidate log: no counterfactual fastest rank.','forward_envelope includes scheduling gaps, not exclusive compute time.'])
 (OUT/'p-tail-summary.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:result[k] for k in ['all','tails','tail_binding_sessions','tail_root_sessions','by_rank','tail_buckets']},indent=2))

def plots():
 # Use the system NumPy/matplotlib pair; the environment's NumPy 2 is ABI-incompatible.
 sys.path.insert(0,'/usr/lib/python3/dist-packages')
 spec=importlib.util.spec_from_file_location('routing',REPO/'p8d8-adaptive-31625-20260923/scripts/analyze_session_routing.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 cases=json.loads((OUT/'routing-summary.json').read_text())
 for label,c in cases.items():
  c['rows']=list(csv.DictReader((OUT/f'{label}-request-routing.csv').open()))
  for r in c['rows']:
   for k in ['turn','start_ns','p_rank','d_rank','input_tokens','output_tokens','device_tokens','host_tokens','miss_tokens']:r[k]=int(float(r[k])) if k=='output_tokens' else int(r[k])
   for k in ['ttft_ms','p_queue_ms']:r[k]=float(r[k])
 mod.figures(cases,OUT);mod.interactive(cases,OUT)
 extra={}
 for label,c in cases.items():
  mult=[s for s in c['sessions'] if s['requests']>=2]
  extra[label]=dict(root_sessions=len(c['sessions']),root_multi_rank=sum(s['ranks_used']>1 for s in mult),root_multi_request=len(mult),mean_ranks_per_multi_root=statistics.fmean(s['ranks_used'] for s in mult),miss_fraction=sum(r['miss_tokens'] for r in c['rows'])/sum(r['input_tokens'] for r in c['rows']))
 (OUT/'root-and-cache-summary.json').write_text(json.dumps(extra,indent=2)+'\n')
 print(json.dumps(extra,indent=2))

def blocking_windows():
 records=[]
 for line in (RUN/'analysis/joined-requests.jsonl').open():
  x=json.loads(line);p=x.get('prefill')
  if not p:continue
  t=p['times'];off=(p['wall_ns']-p['mono_ns'])/1e9;m=x['metadata']
  records.append(dict(rid=x['rid'],phase=x['phase'],session=m['x_correlation_id'],conversation=m['conversation_id'],turn=m['turn_index'],rank=p['dp_rank'],input=p['input_tokens'],gpu=p['cached_device'],host=p['cached_host'],miss=p['input_tokens']-sum(p.get(k,0) for k in ['cached_device','cached_host','cached_storage']),queue_s=p['durations_ms']['queue_ms']/1000,forward_s=p['durations_ms']['forward_envelope_ms']/1000,queue_start=off+t['wait_queue_entry_time'],forward_start=off+t['forward_entry_time'],forward_end=off+t['prefill_finished_time']))
 tails=sorted([r for r in records if r['phase']=='profiling' and r['queue_s']>10],key=lambda r:-r['queue_s'])
 long=[r for r in records if r['forward_s']>30]
 matches=[]
 for r in tails:
  candidates=[(max(0,min(r['forward_start'],b['forward_end'])-max(r['queue_start'],b['forward_start'])),b) for b in long if b['rank']==r['rank'] and b['rid']!=r['rid']]
  overlap,b=max(candidates,key=lambda x:x[0]) if candidates else (0,None)
  matches.append(dict(tail=r,overlap_seconds=overlap,queue_overlap_fraction=overlap/r['queue_s'],overlapping_long_prefill=b if overlap else None))
 result=dict(definition='Same-rank forward envelope >30s overlapping observed queue interval; temporal association, not exclusive GPU occupancy or proven causality.',tail_count=len(tails),overlap_ge80pct=sum(m['queue_overlap_fraction']>=.8 for m in matches),overlap_ge80pct_queue_seconds=sum(m['tail']['queue_s'] for m in matches if m['queue_overlap_fraction']>=.8),matches=matches)
 (OUT/'tail-blocking-windows.json').write_text(json.dumps(result,indent=2)+'\n')
 print('overlap',result['overlap_ge80pct'],'of',len(tails))
 # Three distinct episodes, with relative wall time and measured stage boundaries.
 import matplotlib.pyplot as plt
 chosen=[];seen=set()
 for m in matches:
  b=m['overlapping_long_prefill']
  if b and b['rid'] not in seen:
   chosen.append(m);seen.add(b['rid'])
  if len(chosen)==3:break
 fig,axes=plt.subplots(3,1,figsize=(12,7),constrained_layout=True)
 for ax,m in zip(axes,chosen):
  b=m['overlapping_long_prefill'];r=m['tail'];base=b['forward_start']
  ax.barh(1,b['forward_s'],left=0,color='#ca7631',label='Long request: forward envelope')
  ax.barh(0,r['queue_s'],left=r['queue_start']-base,color='#8e9cab',label='Following request: queue')
  ax.barh(0,r['forward_s'],left=r['forward_start']-base,color='#2879aa',label='Following request: forward envelope')
  ax.set(yticks=[0,1],yticklabels=[f"tail: miss {r['miss']:,}",f"long: miss {b['miss']:,}"],xlabel='Seconds since long request entered forward',title=f"P{r['rank']}: queue {r['queue_s']:.1f}s; long input {b['input']:,} tokens")
 axes[0].legend(fontsize=8,loc='upper right')
 fig.suptitle('Same-rank long Prefill overlaps short-miss request queues (observed timing)')
 fig.savefig(OUT/'tail-blocking-windows.png',dpi=160);fig.savefig(OUT/'tail-blocking-windows.pdf');plt.close(fig)

if __name__=='__main__':audit();plots();blocking_windows()
