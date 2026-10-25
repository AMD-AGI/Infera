# Decode trace analysis for llr7r (yihou). Usage: python3 -I dtrace.yihou.py <trace.json.gz> <out_prefix>
import gzip,json,sys,collections,bisect,re,statistics as S
f,outp=sys.argv[1],sys.argv[2]
d=json.load(gzip.open(f)); ev=d['traceEvents']
X=[e for e in ev if e.get('ph')=='X']
rt={e['args'].get('correlation'):e for e in X if e.get('cat')=='cuda_runtime'}
ks=sorted([e for e in X if e.get('cat') in('kernel','gpu_memcpy')],key=lambda e:e['ts'])
cpu_ua=sorted([e for e in X if e.get('cat')=='user_annotation' and e['tid']==min(x['tid'] for x in X if x.get('cat')=='cuda_runtime')],key=lambda e:e['ts'])
mtid=cpu_ua[0]['tid']
gua=sorted([e for e in X if e.get('cat')=='gpu_user_annotation'],key=lambda e:e['ts'])
def pct(a,p):
    a=sorted(a); 
    return a[min(len(a)-1,int(p/100*len(a)))] if a else float('nan')
# --- phase of each kernel: enclosing innermost CPU user annotation of its launch call
PH={'draft':'draft','draft_extend':'draft_extend','copy_result_to_cpu':'copy_result'}
def phase_of(ts):
    best=None
    for u in cpu_ua:  # small list
        if u['ts']<=ts<=u['ts']+u['dur']:
            if best is None or u['dur']<best['dur']: best=u
    if best is None: return 'outside'
    n=best['name']
    if n.startswith('step[VERIFY'): return 'verify'
    return PH.get(n,n)
for k in ks:
    r=rt.get(k['args'].get('correlation'))
    k['_launch']=r['name'] if r else None
    k['_ph']=phase_of(r['ts']) if r else 'nolaunch'
# --- categories
CAT=[('indexer',r'mqa_logits|dsa_topk|coop_topk|indexer|hadamard|_fused_dsa|topk_v2|dsa_index|index_k|fp8_index'),
 ('sparse_mla',r'sparse_mla|mla_decode|mla_reduce|mla_fwd|_fused_qk_rope_cat_and_cache_mla'),
 ('moe',r'moe|MoeSorting|fused_append_shared_experts|grouped_topk|biased_grouped|topk_softmax|topk_sigmoid'),
 ('allreduce_comm',r'cross_device_reduce|nccl|rccl|all_gather|allreduce|all_reduce|reduce_scatter'),
 ('gemm_dense',r'^Cijk_|hgemm|_gemm_a16|gemm_a8w8|gemm_afp4|gemm_a4w4|f4gemm|_gemm|bpreshuffle|gemm'),
 ('norm_rope_act_quant',r'rmsnorm|layernorm|Layernorm|rope|act_and_mul|silu|gelu|act_quant|quant|kn_entry_2c_sbhd|dynamic_per'),
 ('sampling_spec',r'argmax|ArgMax|VerifyTree|_verify_commit|_draft_topk|topk_reg|softmax|sampl|_prepare_draft|assign_draft|draft'),
 ('memcpy_memset',r'Memcpy|memcpy|copyBuffer|FillFunctor|direct_copy|_small_copy|CatArray|fill|Memset|gather|index_elementwise|indexSelect|scan|bfloat16tofloat32|copy'),
]
CR=[(c,re.compile(p)) for c,p in CAT]
def cat(n):
    for c,r in CR:
        if r.search(n): return c
    return 'other'
def short(n):
    n=n.replace('void ','')
    return n[:110]
# --- steps from GPU annotation scheduler.run_batch
steps=[g for g in gua if g['name']=='scheduler.run_batch']
# kernels to steps by kernel start within [step.ts, next step.ts)
starts=[s['ts'] for s in steps]
res={'file':f,'nsteps_gpu_annot':len(steps)}
# union busy
def union(iv):
    iv=sorted(iv); tot=0; cs=None; ce=None
    for a,b in iv:
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: tot+=ce-cs; cs,ce=a,b
    if cs is not None: tot+=ce-cs
    return tot
# batch size per step from CPU annotation step[VERIFY bs=N]: map via the verify launch
vb=[(u['ts'],int(re.search(r'bs=(\d+)',u['name']).group(1))) for u in cpu_ua if u['name'].startswith('step[VERIFY')]
rows=[]
allk_inside=[]
for i,s in enumerate(steps[:-1]):
    a,b=s['ts'],steps[i+1]['ts']
    kk=[k for k in ks if a<=k['ts']<b]
    if not kk: continue
    first=kk[0]['ts']; last=max(k['ts']+k['dur'] for k in kk)
    busy=union([(k['ts'],k['ts']+k['dur']) for k in kk])
    ph=collections.defaultdict(float); phspan={}
    for k in kk:
        ph[k['_ph']]+=k['dur']
        p=phspan.setdefault(k['_ph'],[k['ts'],k['ts']+k['dur']]); p[0]=min(p[0],k['ts']); p[1]=max(p[1],k['ts']+k['dur'])
    # bs: verify kernels' launch time -> nearest preceding verify annot
    vk=[k for k in kk if k['_ph']=='verify']
    bs=None
    if vk:
        lt=rt[vk[0]['args']['correlation']]['ts']
        cand=[x for x in vb if x[0]<=lt]; bs=cand[-1][1] if cand else None
    # gaps inside phases boundaries
    gaps={}
    order=sorted(phspan.items(),key=lambda x:x[1][0])
    for j in range(len(order)-1):
        gaps[order[j][0]+'->'+order[j+1][0]]=(order[j+1][1][0]-order[j][1][1])/1e3
    rows.append(dict(i=i,t=a,period=(b-a)/1e3,first_off=(first-a)/1e3,span=(last-first)/1e3,busy=busy/1e3,nk=len(kk),
        ph={k:v/1e3 for k,v in ph.items()},phspan={k:(v[1]-v[0])/1e3 for k,v in phspan.items()},gaps=gaps,bs=bs))
    allk_inside+=kk
res['steps']=rows
P=[r['period'] for r in rows]; B=[r['busy'] for r in rows]
tot_period=sum(P); tot_busy=sum(B)
# --- name table
byname=collections.defaultdict(lambda:[0,0.0,collections.Counter()])
bycat=collections.defaultdict(float); bycatph=collections.defaultdict(float)
for k in allk_inside:
    n=k['name']; e=byname[n]; e[0]+=1; e[1]+=k['dur']; e[2][k['_ph']]+=k['dur']
    c=cat(n); bycat[c]+=k['dur']; bycatph[(c,k['_ph'])]+=k['dur']
nst=len(rows)
# --- AR durations
ar=collections.defaultdict(list)
for k in allk_inside:
    if 'cross_device_reduce' in k['name']: ar['custom_ar']+= [k['dur']]
    elif 'nccl' in k['name'].lower(): ar['rccl:'+k['_ph']]+=[k['dur']]
# --- long blocking CPU syncs
syncs=[]
pyf=[e for e in X if e.get('cat')=='python_function' and e['tid']==mtid]
pyf.sort(key=lambda e:e['ts'])
for r in X:
    if r.get('cat')=='cuda_runtime' and r['name'] in('hipMemcpyWithStream','hipEventSynchronize','hipStreamSynchronize','hipDeviceSynchronize') and r['dur']>1000:
        enc=[p for p in pyf if p['ts']<=r['ts'] and p['ts']+p['dur']>=r['ts']+r['dur']]
        enc.sort(key=lambda p:-p['dur'])
        names=[p['name'] for p in enc if 'sglang' in p['name']][-4:]
        ua=[u['name'] for u in cpu_ua if u['ts']<=r['ts']<=u['ts']+u['dur']]
        syncs.append((r['ts'],r['dur']/1e3,r['name'],ua,names))
out=open(outp+'.txt','w')
def w(*a): print(*a,file=out)
w('file',f); w('steps',nst,'kernel-events in steps',len(allk_inside))
w('step period ms p10/p50/p90/mean: %.2f %.2f %.2f %.2f'%(pct(P,10),pct(P,50),pct(P,90),S.mean(P)))
w('GPU busy (union) per step ms p50/p90/mean: %.2f %.2f %.2f ; busy frac %.4f ; idle per step mean %.3f ms'%(pct(B,50),pct(B,90),S.mean(B),tot_busy/tot_period,(tot_period-tot_busy)/nst))
w('sum kernel dur / period (overlap>1 means multi-stream) %.4f'%(sum(k['dur'] for k in allk_inside)/1e3/tot_period))
bss=collections.Counter(r['bs'] for r in rows); w('verify bs per step',dict(bss))
# phase
phs=collections.defaultdict(list)
for r in rows:
    for k,v in r['ph'].items(): phs[k].append(v)
w('\nphase kernel-time per step (ms): mean p50 p90  [n steps]')
for k,v in sorted(phs.items(),key=lambda x:-sum(x[1])): w('  %-14s %.3f %.3f %.3f [%d]'%(k,S.mean(v),pct(v,50),pct(v,90),len(v)))
sp=collections.defaultdict(list)
for r in rows:
    for k,v in r['phspan'].items(): sp[k].append(v)
w('phase wall span per step (first kernel start..last end) mean p50:')
for k,v in sp.items(): w('  %-14s %.3f %.3f'%(k,S.mean(v),pct(v,50)))
gg=collections.defaultdict(list)
for r in rows:
    for k,v in r['gaps'].items(): gg[k].append(v)
w('inter-phase GPU gaps (ms; next phase first kernel - prev phase last end) mean p50 p90 n:')
for k,v in gg.items(): w('  %-30s %.3f %.3f %.3f %d'%(k,S.mean(v),pct(v,50),pct(v,90),len(v)))
# between-step gap: last kernel of step i to first kernel of step i+1
bg=[]
for j in range(len(rows)-1):
    a=rows[j]; b=rows[j+1]
    bg.append((b['t']+b['first_off']*1e3-(a['t']+(a['first_off']+a['span'])*1e3))/1e3)
w('between-step GPU gap ms p50/p90/max/mean: %.3f %.3f %.3f %.3f'%(pct(bg,50),pct(bg,90),max(bg),S.mean(bg)))
w('\ncategory per step ms (%% of summed kernel time) ; split by phase')
tk=sum(bycat.values())
for c,v in sorted(bycat.items(),key=lambda x:-x[1]):
    phx=' '.join('%s=%.2f'%(p,bycatph[(c,p)]/1e3/nst) for p in ('verify','draft','draft_extend','outside','copy_result') if bycatph[(c,p)])
    w('  %-20s %7.3f ms %5.1f%%   %s'%(c,v/1e3/nst,100*v/tk,phx))
w('\ntop-30 kernels by total time: ms/step  count/step  %%  cat  phases(ms/step)  name')
for n,(c,t,phc) in sorted(byname.items(),key=lambda x:-x[1][1])[:30]:
    w('  %7.3f %6.1f %5.1f%% %-18s %-40s %s'%(t/1e3/nst,c/nst,100*t/tk,cat(n),' '.join('%s=%.2f'%(p,v/1e3/nst) for p,v in phc.most_common(3)),short(n)))
w('\nall kernel names in "other" category:')
for n,(c,t,phc) in sorted(byname.items(),key=lambda x:-x[1][1]):
    if cat(n)=='other': w('  %7.3f %s'%(t/1e3/nst,short(n)))
w('\nAR / comm kernel durations (us): n/step p10 p50 p90 max sum/step(ms)')
for k,v in ar.items(): w('  %-24s %5.1f %7.1f %7.1f %7.1f %8.1f %.3f'%(k,len(v)/nst,pct(v,10),pct(v,50),pct(v,90),max(v),sum(v)/1e3/nst))
w('\nlong CPU syncs (>1 ms): count %d, total %.1f ms'%(len(syncs),sum(s[1] for s in syncs)))
cs=collections.Counter((s[2],tuple(s[3][-1:]),tuple(n.split('/')[-1] for n in s[4][-2:])) for s in syncs)
for k,v in cs.most_common(10): w('  %3d %s'%(v,k))
w('\nper-step rows: i period busy bs verify draft draft_extend gaps')
for r in rows: w('  %2d %.2f %.2f %s %s %s'%(r['i'],r['period'],r['busy'],r['bs'],' '.join('%s=%.2f'%(k,v) for k,v in r['ph'].items()),' '.join('%s=%.2f'%(k,v) for k,v in r['gaps'].items())))
out.close()
json.dump({'steps':[(r['t'],r['period'],r['busy'],r['bs'],r['ph']) for r in rows],'ar':{k:v for k,v in ar.items()},
  'cat':{c:v/1e3/nst for c,v in bycat.items()},'names':{short(n):[c/nst,t/1e3/nst,cat(n)] for n,(c,t,_) in byname.items()}},open(outp+'.json','w'))
print(open(outp+'.txt').read())
