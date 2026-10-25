# Cross-stage pipeline analysis from ex2.pp{0..3}.json. Usage: combine.yihou.py <dir>
import json, sys, collections, statistics, datetime, re
D=sys.argv[1]; S=[json.load(open(f'{D}/ex2.pp{i}.json')) for i in range(4)]
LAYERS=[20,20,20,18]
utc=lambda t: datetime.datetime.fromtimestamp(t/1e6, datetime.timezone.utc).strftime('%H:%M:%S.%f')[:-3]
def pct(v,q):
    v=sorted(v); return v[min(len(v)-1,int(round(q*(len(v)-1))))] if v else float('nan')
def union(iv):
    iv=sorted(iv); tot=0; cs=ce=None
    for a,b in iv:
        if cs is None: cs,ce=a,b
        elif a<=ce: ce=max(ce,b)
        else: tot+=ce-cs; cs,ce=a,b
    if cs is not None: tot+=ce-cs
    return tot
def ov(iv,a,b): return union([(max(x,a),min(y,b)) for x,y in iv if x<b and y>a])
# batches per stage
B=[]
for k,s in enumerate(S):
    st=s['steps']; bl=[]
    i=0
    while i<len(st):
        e=st[i]; tgt=(e['t'],e['t']+e['d']); dr=None
        if i+1<len(st) and 'DRAFT' in st[i+1]['n'] and st[i+1]['t']-e['t']<1e6: dr=(st[i+1]['t'],st[i+1]['t']+st[i+1]['d']); i+=1
        bl.append(dict(n=e['n'],t0=tgt[0],t1=tgt[1],dr=dr,end=max(tgt[1],dr[1]) if dr else tgt[1])); i+=1
    B.append(bl)
print('batches per stage',[len(b) for b in B]); assert all(B[k][j]['n']==B[0][j]['n'] for k in range(4) for j in range(20))
# long kernels
print('\n## kernels > 30 ms (all stages)')
for k,s in enumerate(S):
    for g in s['G']:
        if g['d']>3e4 and not g['k'].startswith('memcpy') and g['k']!='nccl': print('  PP%d %s dur %7.1f ms s%s %s %s'%(k,utc(g['t']),g['d']/1e3,g['s'],g['k'],g['n']))
# clean batches: all stage windows before 08:52:06.8 stall and no long non-nccl non-memcpy kernel inside any stage's step
def has_long(k,b):
    return any(g['d']>3e4 and g['k']!='nccl' and not g['k'].startswith('memcpy') and B[k][b]['t0']<=g['t']<B[k][b]['end'] for g in S[k]['G'])
stall_t=min(B[0][19]['t0'],min(b['t0'] for k in range(4) for b in B[k] if b['t0']-B[k][max(0,B[k].index(b)-1)]['end']>3e6) if any(b['t0']-B[k][max(0,B[k].index(b)-1)]['end']>3e6 for k in range(4) for b in B[k]) else 1e30)
print('\nfirst stall-affected batch start >=',utc(stall_t))
clean=[j for j in range(20) if all(B[k][j]['end']<stall_t+1 and not has_long(k,j) for k in range(4)) and j>0]
print('clean batches (no >30ms anomalous kernel in any stage, before stall):',clean)
# timeline table
print('\n## per-batch timeline: per stage GPU start(rel PP0 batch0) / GPU dur (PP3: target+draft) ; handoff k->k+1 = start(k+1)-end(k)')
T0=B[0][0]['t0']
for j in range(20):
    row=[]
    for k in range(4):
        b=B[k][j]; row.append('%8.1f/%6.1f%s'%((b['t0']-T0)/1e3,(b['end']-b['t0'])/1e3,'' if not b['dr'] else '(d%.1f)'%((b['dr'][1]-b['dr'][0])/1e3)))
    ho=['%6.1f'%((B[k+1][j]['t0']-B[k][j]['end'])/1e3) for k in range(3)]
    print('  b%02d %-28s %s | handoff %s %s'%(j,B[0][j]['n'],'  '.join(row),' '.join(ho),'CLEAN' if j in clean else ''))
# stage period & handoff stats over clean
print('\n## clean-batch stats per stage')
res={}
for k in range(4):
    gd=[(B[k][j]['end']-B[k][j]['t0'])/1e3 for j in clean]
    tg=[(B[k][j]['t1']-B[k][j]['t0'])/1e3 for j in clean]
    dr=[(B[k][j]['dr'][1]-B[k][j]['dr'][0])/1e3 for j in clean if B[k][j]['dr']]
    per=[(B[k][j]['t0']-B[k][j-1]['t0'])/1e3 for j in clean if j-1>=0]
    gap=[(B[k][j]['t0']-B[k][j-1]['end'])/1e3 for j in clean]
    print('  PP%d (%d layers): step GPU (target%s) p50 %.1f p90 %.1f mean %.1f ms | per layer %.2f ms | draft p50 %s | start-to-start period p50 %.1f mean %.1f | gap before step p50 %.1f mean %.1f'%(k,LAYERS[k],'+draft' if dr else '',pct(gd,.5),pct(gd,.9),statistics.mean(gd),statistics.mean(tg)/LAYERS[k],'%.1f'%pct(dr,.5) if dr else '-',pct(per,.5),statistics.mean(per),pct(gap,.5),statistics.mean(gap)))
for k in range(3):
    h=[(B[k+1][j]['t0']-B[k][j]['end'])/1e3 for j in clean]; print('  handoff PP%d->PP%d (next stage GPU start - this stage GPU end), clean: p50 %.1f p90 %.1f min %.1f ms'%(k,k+1,pct(h,.5),pct(h,.9),min(h)))
# all-batch step stats
print('\n## all 20 batches: step GPU p50/p90 per stage')
for k in range(4):
    gd=[(b['end']-b['t0'])/1e3 for b in B[k]]; print('  PP%d p50 %.1f p90 %.1f mean %.1f max %.1f'%(k,pct(gd,.5),pct(gd,.9),statistics.mean(gd),max(gd)))
# window busy fractions: full active window (b0 start..b19 end excluding stalls) and clean window (first clean batch start .. last clean batch end)
print('\n## GPU busy fractions')
for k in range(4):
    G=S[k]['G']
    segs=[]; 
    for j in range(20):
        a=B[k][j-1]['end'] if j>0 else B[k][0]['t0']; b=B[k][j]['end']
        if B[k][j]['t0']-a>3e6: a=B[k][j]['t0']
        segs.append((a,b))
    act=union(segs)
    def bus(filt,segs_):
        iv=[(g['t'],g['t']+g['d']) for g in G if filt(g)]
        return sum(ov(iv,a,b) for a,b in segs_)
    cw=[(B[k][clean[0]-1]['end'],B[k][clean[-1]]['end'])]; cact=cw[0][1]-cw[0][0]
    fk=lambda g: g['k']!='nccl' and not g['k'].startswith('memcpy') and g['k']!='memset'
    fa=lambda g: True
    fn=lambda g: g['k']=='nccl'
    print('  PP%d active %.0f ms: compute kernels %.1f%%, any(incl memcpy,nccl) %.1f%%, nccl %.1f%% | clean window %.0f ms: compute kernels %.1f%%, any %.1f%%, nccl %.1f%%'%(k,act/1e3,100*bus(fk,segs)/act,100*bus(fa,segs)/act,100*bus(fn,segs)/act,cact/1e3,100*bus(fk,cw)/cact,100*bus(fa,cw)/cact,100*bus(fn,cw)/cact))
# per-step category breakdown on clean batches (kernels starting inside step window)
print('\n## per-batch kernel time by category (clean batches, mean ms per batch; sum of kernel durations launched in fwd/draft ctx)')
cats=sorted({g['k'] for s in S for g in s['G']})
tab={}
for k in range(4):
    acc=collections.defaultdict(float)
    for j in clean:
        b=B[k][j]
        for g in S[k]['G']:
            if b['t0']<=g['t']<b['end'] and g['x'] in('fwd','draft'):
                acc[(g['x'],g['k'])]+=g['d']/1e3/len(clean)
    tab[k]=acc
keys=sorted({x for k in tab for x in tab[k]},key=lambda x:-sum(tab[k].get(x,0) for k in tab))
print('  %-48s %9s %9s %9s %9s'%('ctx/category','PP0','PP1','PP2','PP3'))
for x in keys:
    if max(tab[k].get(x,0) for k in tab)<0.3: continue
    print('  %-48s %9.2f %9.2f %9.2f %9.2f'%(x,*[tab[k].get(x,0) for k in range(4)]))
print('  %-48s %9.2f %9.2f %9.2f %9.2f'%('TOTAL',*[sum(tab[k].values()) for k in range(4)]))
# gap attribution (clean): main-thread python + nccl by ctx
print('\n## between-step gaps on each stage (clean batches): union overlap ms per gap (mean)')
names=['_pp_recv_dict_from_prev_stage','_pp_recv_pyobj_from_prev_stage','_pp_recv_proxy_tensors','recv_requests','synchronize','_pp_commit_send_output_work_and_preprocess_output_tensors','_pp_send_dict_to_next_stage','process_batch_result_disagg_prefill','send_kv_chunk','checkpoint_disagg_prefill','process_disagg_prefill_inflight_queue','_execute_kv_backup','load_back','run_batch','resolve_forward_inputs']
for k in range(4):
    PM=S[k]['PM']; G=S[k]['G']; acc=collections.defaultdict(float); tg=0
    for j in clean:
        a=B[k][j-1]['end']; b=B[k][j]['t0']
        if b<=a: continue
        tg+=b-a
        for n in names: acc['cpu:'+n]+=ov([(p['t'],p['t']+p['d']) for p in PM if p['n']==n],a,b)
        for x in ('pp_send','pp_recv'): acc['gpu nccl '+x]+=ov([(g['t'],g['t']+g['d']) for g in G if g['k']=='nccl' and g['x']==x],a,b)
        acc['gpu compute kernels']+=ov([(g['t'],g['t']+g['d']) for g in G if g['k']!='nccl' and not g['k'].startswith('memcpy')],a,b)
        acc['gpu memcpy']+=ov([(g['t'],g['t']+g['d']) for g in G if g['k'].startswith('memcpy')],a,b)
    n=len(clean)
    print('  PP%d mean gap %.1f ms: '%(k,tg/1e3/n)+'; '.join('%s %.1f'%(x,v/1e3/n) for x,v in sorted(acc.items(),key=lambda y:-y[1]) if v/1e3/n>=0.5))
# nccl kernel durations by ctx, clean
print('\n## nccl kernels by ctx (clean window), ms per batch and max')
for k in range(4):
    a=B[k][clean[0]-1]['end']; b=B[k][clean[-1]]['end']
    for x in ('pp_send','pp_recv'):
        v=[g['d']/1e3 for g in S[k]['G'] if g['k']=='nccl' and g['x']==x and a<=g['t']<b]
        if v: print('  PP%d %s n=%d sum/batch %.1f ms p50 %.2f p90 %.1f max %.1f'%(k,x,len(v),sum(v)/len(clean),pct(v,.5),pct(v,.9),max(v)))
# memcpy by ctx over whole active (20 batches)
print('\n## memcpy + hicache kernels by ctx/kind over 20 batches (exclude stalls): n, sum ms, bytes, sum ms of events<=1ms, bytes/time of those')
for k in range(4):
    lo=B[k][0]['t0']; hi=B[k][19]['end']
    acc=collections.defaultdict(lambda:[0,0.0,0,0.0,0])
    for g in S[k]['G']:
        if not(lo<=g['t']<=hi): continue
        if any(B[k][j]['t0']-B[k][j-1]['end']>3e6 and B[k][j-1]['end']<=g['t']<B[k][j]['t0'] for j in range(1,20)): continue
        if g['k'].startswith('memcpy') or g['k']=='hicache_kernel':
            r=acc[(g['x'],g['k'])]; r[0]+=1; r[1]+=g['d']/1e3; r[2]+=g['b']
            if g['d']<=1e3: r[3]+=g['d']/1e3; r[4]+=g['b']
    for x,r in sorted(acc.items(),key=lambda y:-y[1][1]):
        if r[1]>=0.5: print('  PP%d %-34s n=%5d sum %8.1f ms  %.3f GB | events<=1ms: %.1f ms %.3f GB'%(k,x,r[0],r[1],r[2]/1e9,r[3],r[4]/1e9))
# long memcpys: same-stream kernel overlap
print('\n## memcpy > 10 ms: fraction overlapped by compute kernels on ANY stream / SAME stream')
for k in range(4):
    G=S[k]['G']
    for g in G:
        if g['k'].startswith('memcpy') and g['d']>1e4:
            a,b=g['t'],g['t']+g['d']
            same=ov([(h['t'],h['t']+h['d']) for h in G if h['s']==g['s'] and h is not g and not h['k'].startswith('memcpy')],a,b)
            anyk=ov([(h['t'],h['t']+h['d']) for h in G if h['k']!='nccl' and not h['k'].startswith('memcpy')],a,b)
            print('  PP%d %s %-14s s%s %7.1f ms %6d B ctx=%-12s anyK %.0f%% sameStreamK %.0f%%'%(k,utc(a),g['k'],g['s'],g['d']/1e3,g['b'],g['x'],100*anyk/(b-a),100*same/(b-a)))
# hicache shapes
print('\n## hicache cpu_op samples (PP0)')
seen=collections.Counter()
for h in S[0]['hs']:
    if seen[h['n']]<3: seen[h['n']]+=1; print('  ',utc(h['t']),h['n'],json.dumps(h['a'])[:600])
# mooncake worker threads
print('\n## Mooncake transfer worker threads (python, union per name over the 20-batch active span excl stalls)')
for k in range(4):
    lo=B[k][0]['t0']; hi=B[k][19]['end']; acc=collections.defaultdict(list)
    for p in S[k]['PO']:
        a,b=max(p['t'],lo),min(p['t']+p['d'],hi)
        if b>a: acc[p['n']].append((a,b))
    print('  PP%d '%k+'; '.join('%s n=%d union %.1f ms'%(n,len(v),union(v)/1e3) for n,v in acc.items()))
