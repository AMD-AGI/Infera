# Prefill PP-stage trace analysis (llr7r). Usage: python3 -I anal.yihou.py <trace.json.gz> <out_prefix>
import gzip, json, sys, collections, re, bisect, statistics, datetime
f=sys.argv[1]; outp=sys.argv[2]
d=json.load(gzip.open(f)); ev=d['traceEvents']; base_ns=d.get('baseTimeNanoseconds',0)
out=open(outp+'.txt','w')
def P(*a):
    s=' '.join(str(x) for x in a); print(s); out.write(s+'\n')
def utc(ts): return datetime.datetime.utcfromtimestamp(base_ns/1e9+ts/1e6).strftime('%H:%M:%S.%f')[:-3]
def pct(v,q):
    if not v: return float('nan')
    v=sorted(v); i=min(len(v)-1,max(0,int(round(q*(len(v)-1))))); return v[i]
X=[e for e in ev if e.get('ph')=='X']
GPU=[e for e in X if e.get('cat') in('kernel','gpu_memcpy','gpu_memset')]
RT={e['args'].get('correlation'):e for e in X if e.get('cat')=='cuda_runtime' and 'correlation' in e.get('args',{})}
lc=collections.Counter(e['tid'] for e in RT.values()); main_tid=lc.most_common(1)[0][0]
P('file',f); P('launch tids',lc.most_common(6),'main',main_tid)
# ---- sweep: innermost cpu_op + python stack for each runtime launch
def sweep(intervals, queries):
    # intervals: list of (start,end,name) ; queries: list of (ts,key) -> dict key->stack(list of names outer->inner)
    intervals.sort(key=lambda x:(x[0],-x[1])); queries.sort()
    res={}; st=[]; i=0
    for ts,key in queries:
        while i<len(intervals) and intervals[i][0]<=ts:
            s,e_,n=intervals[i]; 
            while st and st[-1][1]<=s: st.pop()
            st.append((s,e_,n)); i+=1
        while st and st[-1][1]<ts: st.pop()
        res[key]=[x[2] for x in st if x[0]<=ts<=x[1]]
    return res
bytid=collections.defaultdict(list)
for c in RT.values(): bytid[c['tid']].append((c['ts'],c['args']['correlation']))
pystack={}; opstack={}
for tid,q in bytid.items():
    py=[(e['ts'],e['ts']+e['dur'],e['name']) for e in X if e.get('cat')=='python_function' and e['tid']==tid]
    op=[(e['ts'],e['ts']+e['dur'],e['name']) for e in X if e.get('cat')=='cpu_op' and e['tid']==tid]
    pystack.update(sweep(py,list(q))); opstack.update(sweep(op,list(q)))
CTX=[('draft',re.compile(r'_draft_extend_for_prefill|draft_extend|draft_worker|forward_draft')),
     ('hicache_backup',re.compile(r'backup_from_device|_execute_kv_backup|start_writing|submit_device_to_host')),
     ('hicache_load',re.compile(r'load_back|start_loading|load_to_device|submit_host_to_device')),
     ('kv_send',re.compile(r'send_kv_chunk|maybe_send_cached_prefix|mooncake')),
     ('pp_comm',re.compile(r'_pp_send|_pp_recv|send_tensor_dict|recv_tensor_dict|send_object|recv_object|_pp_commit')),
     ('target_fwd',re.compile(r'forward_batch_generation|run_batch|forward_extend|model_runner')),
     ]
def ctx_of(corr):
    stk=pystack.get(corr); 
    if stk is None: return 'no_launch'
    s=' | '.join(stk)
    for n,r in CTX:
        if r.search(s): return n
    return 'other'
KC=[('attn_sparse_mla',r'_sparse_mla_fwd'),('attn_misc(rope/cache/bmm/qknorm)',r'fused_qk_rope_cat_and_cache_mla|kn_entry_2c_sbhd|_batched_gemm_a8w8|fused_qk_rmsnorm'),
    ('indexer(logits/topk/hadamard/kquant)',r'mqa_logits|topk_packed|hadamard|indexer_k_quant|cp_gather_indexer|act_quant_kernel|topk_transform'),
    ('moe',r'moe|gemm2_a4w4|mxfp4|fused_append_shared_experts|dynamic_per_group_scaled_quant|topk_reg_kernel|act_and_mul'),
    ('dense_gemm',r'hgemm|Cijk_|_gemm_a16_w16'),('norm',r'rmsnorm|Layernorm'),('nccl',r'nccl'),
    ('hicache_kernel',r'transfer_kernel_impl|hicache_transfer'),('rocprim_scan',r'rocprim'),('elementwise/other',r'.')]
KC=[(n,re.compile(r)) for n,r in KC]
def kcat(e):
    if e['cat']=='gpu_memcpy': return 'memcpy_'+e['args'].get('kind','?')
    if e['cat']=='gpu_memset': return 'memset'
    for n,r in KC:
        if r.search(e['name']): return n
# ---- streams
sb=collections.Counter(); 
for e in GPU: sb[e['args'].get('stream')]+=e['dur']
comp_stream=max((s for s in sb), key=lambda s: sum(1 for e in GPU if e['args'].get('stream')==s and e['cat']=='kernel'))
P('streams busy ms',{s:round(v/1e3,1) for s,v in sb.items()},'compute stream',comp_stream)
# ---- steps
steps=[]
for e in X:
    if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('step['):
        m=re.search(r'bs=(\d+) toks=(\d+)',e['name']); steps.append([e['ts'],e['ts']+e['dur'],int(m.group(1)),int(m.group(2)),e['name']])
steps.sort()
cpusteps=sorted((e['ts'],e['dur']) for e in X if e.get('cat')=='user_annotation' and e['name'].startswith('step['))
P('n steps',len(steps))
# stall detection: gaps between consecutive steps > STALL_S
STALL=3e6
gaps=[(steps[i][0]-steps[i-1][1],i) for i in range(1,len(steps))]
stalls=[(steps[i-1][1],steps[i][0]) for g,i in gaps if g>STALL]
win=(steps[0][0],steps[-1][1])
for s,e_ in stalls: P('STALL excluded %s -> %s (%.1f s)'%(utc(s),utc(e_),(e_-s)/1e6))
P('window %s -> %s span %.1f ms, excl stalls %.1f ms'%(utc(win[0]),utc(win[1]),(win[1]-win[0])/1e3,(win[1]-win[0]-sum(b-a for a,b in stalls))/1e3))
def in_win(a,b):
    if b<win[0] or a>win[1]: return False
    for s,e_ in stalls:
        if a>=s and b<=e_: return False
    return True
def union(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for a,b in iv:
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: tot+=ce-cs; cs,ce=a,b
    if cs is not None: tot+=ce-cs
    return tot
def clip(a,b,lo,hi): return (max(a,lo),min(b,hi))
segs=[]; lo=win[0]
for s,e_ in stalls: segs.append((lo,s)); lo=e_
segs.append((lo,win[1])); active=sum(b-a for a,b in segs)
def busy(evs):
    iv=[]
    for e in evs:
        for lo_,hi_ in segs:
            a,b=clip(e['ts'],e['ts']+e['dur'],lo_,hi_)
            if b>a: iv.append((a,b))
    return union(iv)
allb=busy(GPU); kbusy=busy([e for e in GPU if e['cat']=='kernel' and 'nccl' not in e['name']]); compb=busy([e for e in GPU if e['args'].get('stream')==comp_stream])
nonccl=busy([e for e in GPU if 'nccl' not in e['name']])
P('ACTIVE window %.1f ms: GPU busy(any stream, incl memcpy+nccl) %.1f%%, excl nccl %.1f%%, compute-kernels only (no memcpy, no nccl, all streams) %.1f%%, compute stream %s (incl its memcpy) %.1f%%'%(active/1e3,100*allb/active,100*nonccl/active,100*kbusy/active,comp_stream,100*compb/active))
# ---- per-step
P('\n# per step: utc_start gpu_ms bs toks kernel_busy_ms(non-nccl kernels, any stream) gap_before_ms cpu_launch_ms')
gdur=[];prev=None;rows=[];gapl=[]
for i,(a,b,bs,tk,n) in enumerate(steps):
    cb=union([clip(e['ts'],e['ts']+e['dur'],a,b) for e in GPU if e['cat']=='kernel' and 'nccl' not in e['name'] and e['ts']<b and e['ts']+e['dur']>a])
    gap=(a-prev) if prev else float('nan'); prev=b
    st=any(abs(a-e_)<1 for s,e_ in stalls)
    cl=next((du for t,du in cpusteps if t<=a+1e3 and t>=a-5e4),float('nan'))
    rows.append((a,b,bs,tk,cb,gap,st)); 
    P('%s %8.1f bs=%d toks=%d busy=%.1f gap_before=%.1f%s cpu_launch=%.1f'%(utc(a),(b-a)/1e3,bs,tk,cb/1e3,gap/1e3,' (STALL)' if st else '',cl/1e3))
    gdur.append((b-a)/1e3)
    if i>0 and not st: gapl.append(gap/1e3)
P('step GPU ms p50 %.1f p90 %.1f mean %.1f max %.1f; toks/step mean %.0f; non-stall inter-step gap p50 %.1f p90 %.1f mean %.1f ms'%(pct(gdur,.5),pct(gdur,.9),statistics.mean(gdur),max(gdur),statistics.mean(r[3] for r in rows),pct(gapl,.5),pct(gapl,.9),statistics.mean(gapl) if gapl else 0))
in_steps=sum(r[1]-r[0] for r in rows); P('time inside step annotations %.1f ms = %.1f%% of active; gaps %.1f%%'%(in_steps/1e3,100*in_steps/active,100*(1-in_steps/active)))
# ---- category breakdown (active window)
cat=collections.defaultdict(lambda:[0,0.0,0]); ctxc=collections.defaultdict(float); ctxk=collections.defaultdict(lambda:collections.defaultdict(float))
for e in GPU:
    if not in_win(e['ts'],e['ts']+e['dur']): continue
    k=kcat(e); cat[k][0]+=1; cat[k][1]+=e['dur']; cat[k][2]+=e['args'].get('bytes',0) or 0
    c=ctx_of(e['args'].get('correlation')); e['_ctx']=c; ctxc[c]+=e['dur']; ctxk[c][k]+=e['dur']
P('\n# kernel category (sum of durations, active window; per step = /%d)'%len(steps))
for k,v in sorted(cat.items(),key=lambda x:-x[1][1]): P('%-40s n=%6d %9.1f ms  %7.2f ms/step  bytes=%.3f GB'%(k,v[0],v[1]/1e3,v[1]/1e3/len(steps),v[2]/1e9))
P('\n# launch context (python stack) -> GPU ms (top categories)')
for c,v in sorted(ctxc.items(),key=lambda x:-x[1]):
    P('%-16s %9.1f ms  '%(c,v/1e3)+', '.join('%s=%.1f'%(k,x/1e3) for k,x in sorted(ctxk[c].items(),key=lambda y:-y[1])[:6]))
# ---- memcpy detail
P('\n# memcpy/hicache-kernel detail by (kind, stream, cpu_op, python leaf)')
md=collections.defaultdict(lambda:[0,0.0,0,0.0])
for e in GPU:
    if not in_win(e['ts'],e['ts']+e['dur']): continue
    if e['cat']=='gpu_memcpy' or re.search('transfer_kernel_impl|hicache_transfer',e['name']):
        corr=e['args'].get('correlation'); op=(opstack.get(corr) or ['-'])[-1]
        py=[x for x in (pystack.get(corr) or []) if '/sglang/' in x][-2:]
        py=' < '.join(x.split('/sglang/')[-1] for x in reversed(py))
        k=(e['args'].get('kind',e['name'][:30]),e['args'].get('stream'),op,py)
        md[k][0]+=1; md[k][1]+=e['dur']; md[k][2]+=e['args'].get('bytes',0) or 0; md[k][3]=max(md[k][3],e['dur'])
for k,v in sorted(md.items(),key=lambda x:-x[1][1])[:30]: P('n=%5d %8.1f ms max %7.1f ms %8.3f GB %s'%(v[0],v[1]/1e3,v[3]/1e3,v[2]/1e9,k))
# ---- top kernels with cpu op
P('\n# top kernels: name | top cpu_op | ms')
kd=collections.defaultdict(lambda:[0,0.0,collections.Counter()])
for e in GPU:
    if e['cat']!='kernel' or not in_win(e['ts'],e['ts']+e['dur']): continue
    corr=e['args'].get('correlation'); op=(opstack.get(corr) or ['-'])[-1]
    k=e['name'][:70]; kd[k][0]+=1; kd[k][1]+=e['dur']; py=[x for x in (pystack.get(corr) or []) if '/sglang/' in x][-1:]; kd[k][2][op+' @'+e.get('_ctx','?')+' '+(py[0].split('/sglang/')[-1] if py else '')]+=e['dur']
for k,v in sorted(kd.items(),key=lambda x:-x[1][1])[:25]: P('%8.1f ms n=%5d %s | %s'%(v[1]/1e3,v[0],k,'; '.join('%s %.0f'%(o,x/1e3) for o,x in v[2].most_common(2))))
# ---- inter-step gap attribution
P('\n# inter-step gaps (non-stall): what the compute stream / GPU / CPU main thread did')
nccl=[e for e in GPU if 'nccl' in e['name']]
pyann=[e for e in X if e.get('cat')=='python_function' and e['tid']==main_tid and re.search(r'\): (_pp_recv_dict_from_prev_stage|_pp_recv_proxy_tensors|_pp_send_dict_to_next_stage|_pp_recv_pyobj_from_prev_stage|process_batch_result_disagg_prefill|recv_requests|get_next_disagg_prefill_batch\w*|_pp_launch_batch|synchronize|checkpoint_disagg_prefill|process_disagg_prefill_inflight_queue|_pp_commit_send_output\w*|send_kv_chunk|_draft_extend_for_prefill|forward_batch_generation|run_batch|resolve_forward_inputs|kv_to_page_indices|_execute_kv_backup|load_back)$',e['name'])]
pyname=lambda n: n.split('): ')[-1]
agg=collections.defaultdict(float); aggg=collections.defaultdict(float); tg=0; ng=0
for i in range(1,len(rows)):
    if rows[i][6]: continue
    a,b=rows[i-1][1],rows[i][0]; 
    if b<=a: continue
    tg+=b-a; ng+=1
    aggg['gpu_any_busy']+=union([clip(e['ts'],e['ts']+e['dur'],a,b) for e in GPU if e['ts']<b and e['ts']+e['dur']>a])
    aggg['nccl_kernels']+=union([clip(e['ts'],e['ts']+e['dur'],a,b) for e in nccl if e['ts']<b and e['ts']+e['dur']>a])
    aggg['compute_stream_busy']+=union([clip(e['ts'],e['ts']+e['dur'],a,b) for e in GPU if e['args'].get('stream')==comp_stream and e['ts']<b and e['ts']+e['dur']>a])
    by=collections.defaultdict(list)
    for e in pyann:
        if e['ts']<b and e['ts']+e['dur']>a: by[pyname(e['name'])].append(clip(e['ts'],e['ts']+e['dur'],a,b))
    for n,iv in by.items(): agg[n]+=union(iv)
if ng:
    P('n gaps %d total %.1f ms mean %.1f ms'%(ng,tg/1e3,tg/1e3/ng))
    for k,v in aggg.items(): P('  GPU %-22s %8.1f ms (%.0f%% of gap time)'%(k,v/1e3,100*v/tg))
    for k,v in sorted(agg.items(),key=lambda x:-x[1]): P('  CPU main %-55s %8.1f ms (%.0f%%)'%(k,v/1e3,100*v/tg))
# also: CPU main-thread time in pp recv (whole active window, excl stalls)
P('\n# main-thread python totals in active window (union, ms)')
tot=collections.defaultdict(list)
for e in pyann:
    for lo_,hi_ in segs:
        a,b=clip(e['ts'],e['ts']+e['dur'],lo_,hi_)
        if b>a: tot[pyname(e['name'])].append((a,b))
for k,v in sorted(tot.items(),key=lambda x:-union(x[1])): P('  %-55s %9.1f ms'%(k,union(v)/1e3))
# nccl kernels by annotation (send/recv), durations
P('\n# nccl kernels in active window')
na=collections.defaultdict(list)
gann=[e for e in X if e.get('cat')=='gpu_user_annotation' and e['name'].startswith('nccl')]
for e in nccl:
    if not in_win(e['ts'],e['ts']+e['dur']): continue
    st=e['args'].get('stream'); na[st].append(e['dur']/1e3)
for s,v in na.items(): P('  stream %s n=%d sum %.1f ms p50 %.2f p90 %.2f max %.1f'%(s,len(v),sum(v),pct(v,.5),pct(v,.9),max(v)))
json.dump({'steps':[(utc(r[0]),(r[1]-r[0])/1e3,r[2],r[3],r[4]/1e3,r[5]/1e3,r[6]) for r in rows],'active_ms':active/1e3,'busy_any':allb/active,'busy_compute':compb/active,'cat':{k:v[1]/1e3 for k,v in cat.items()},'ctx':{k:v/1e3 for k,v in ctxc.items()}},open(outp+'.json','w'),indent=1)

# ---- long memcpys (>10 ms) and concurrent nccl kernels
P('\n# memcpy events > 10 ms: utc start, dur, kind, stream, bytes, py leaf, overlapping nccl (stream, start, dur)')
t0=win[0]
for e in sorted(GPU,key=lambda e:e['ts']):
    if e['cat']!='gpu_memcpy' or e['dur']<1e4 or not in_win(e['ts'],e['ts']+e['dur']): continue
    a,b=e['ts'],e['ts']+e['dur']; corr=e['args'].get('correlation')
    py=[x for x in (pystack.get(corr) or []) if '/sglang/' in x][-1:]
    ov=[(x['args'].get('stream'),round((x['ts']-t0)/1e3,1),round(x['dur']/1e3,1)) for x in nccl if x['ts']<b and x['ts']+x['dur']>a and x['dur']>1e3]
    rt=RT.get(corr); 
    P('  t=%8.1f %s dur %7.1f %s s%s %d B %s | launch cpu dur %.1f | nccl %s'%((a-t0)/1e3,utc(a),e['dur']/1e3,e['args'].get('kind'),e['args'].get('stream'),e['args'].get('bytes',0),py[0].split('/sglang/')[-1] if py else '-',(rt['dur']/1e3 if rt else -1),ov))
# memcpy duration distribution per (kind, py leaf) for top groups
P('\n# memcpy per-event duration p50/p90 by group')
grp=collections.defaultdict(list)
for e in GPU:
    if e['cat']!='gpu_memcpy' or not in_win(e['ts'],e['ts']+e['dur']): continue
    corr=e['args'].get('correlation'); py=[x for x in (pystack.get(corr) or []) if '/sglang/' in x][-1:]
    grp[(e['args'].get('kind'),py[0].split('/sglang/')[-1] if py else '-')].append((e['dur']/1e3,e['args'].get('bytes',0)))
for k,v in sorted(grp.items(),key=lambda x:-sum(y[0] for y in x[1]))[:10]:
    ds=[y[0] for y in v]; P('  %s n=%d sum %.1f p50 %.3f p90 %.3f max %.1f ms, bytes/ev p50 %d, eff BW(sum bytes/sum time of events<p90) %.1f GB/s'%(k,len(v),sum(ds),pct(ds,.5),pct(ds,.9),max(ds),pct([y[1] for y in v],.5), sum(y[1] for y in v if y[0]<=pct(ds,.9))/max(1e-9,sum(y[0] for y in v if y[0]<=pct(ds,.9)))/1e6))
# compute-stream kernel time outside step annotations, by context
P('\n# non-nccl kernel time OUTSIDE step annotations by context/category')
oc=collections.defaultdict(float)
for e in GPU:
    if e['cat']!='kernel' or 'nccl' in e['name'] or not in_win(e['ts'],e['ts']+e['dur']): continue
    if any(r[0]<=e['ts']<r[1] for r in rows): continue
    oc[(e.get('_ctx'),kcat(e))]+=e['dur']
for k,v in sorted(oc.items(),key=lambda x:-x[1])[:12]: P('  %-50s %8.1f ms'%(k,v/1e3))
