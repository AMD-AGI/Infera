# Per-step decode timeline (yihou). The verify cuda-graph launches have 0 kernel records in these traces,
# so verify is measured as a window: draft GPU end -> draft_extend GPU start.
# Usage: python3 -I dstep.yihou.py <out_prefix> <trace1.json.gz> ...
import gzip,json,sys,collections,re,statistics as S
outp=sys.argv[1]; files=sys.argv[2:]
def pct(a,p):
    a=sorted(a); return a[min(len(a)-1,int(p/100*len(a)))] if a else float('nan')
allr={}
out=open(outp+'.txt','w')
def w(*a): print(*a,file=out); print(*a)
for f in files:
    rank=re.search(r'TP-(\d+)',f).group(1)
    d=json.load(gzip.open(f)); ev=d['traceEvents']; X=[e for e in ev if e.get('ph')=='X']
    rts=[e for e in X if e.get('cat')=='cuda_runtime']; mtid=collections.Counter(e['tid'] for e in rts).most_common(1)[0][0]
    kbyc=collections.defaultdict(list)
    for e in X:
        if e.get('cat') in('kernel','gpu_memcpy'): kbyc[e['args'].get('correlation')].append(e)
    ua=sorted([e for e in X if e.get('cat')=='user_annotation' and e['tid']==mtid],key=lambda e:e['ts'])
    rb=[u for u in ua if u['name']=='scheduler.run_batch']
    gl=sorted([e for e in rts if e['name']=='hipGraphLaunch' and e['tid']==mtid],key=lambda e:e['ts'])
    syn=sorted([e for e in rts if e['name']=='hipMemcpyWithStream' and e['tid']==mtid],key=lambda e:e['ts'])
    # all launches (non-graph too) in window for draft / draft_extend annotation
    rows=[]
    for r in rb:
        a,b=r['ts'],r['ts']+r['dur']
        inside=[u for u in ua if a<=u['ts']<=b]
        v=[u for u in inside if u['name'].startswith('step[VERIFY')]
        dr=[u for u in inside if u['name']=='draft']; de=[u for u in inside if u['name']=='draft_extend']
        if not(v and dr and de): continue
        def kern(u):
            ks=[]
            for e in rts:
                pass
            return ks
        def kernels_in(u):
            ks=[]
            for g in rts_by_ts_range(u['ts'],u['ts']+u['dur']):
                ks+=kbyc.get(g['args'].get('correlation'),[])
            return ks
        rows.append(dict(rb=r,v=v[0],dr=dr[0],de=de[0]))
    # index runtime by ts for range queries
    rts_m=sorted([e for e in rts if e['tid']==mtid],key=lambda e:e['ts']); tsl=[e['ts'] for e in rts_m]
    import bisect
    def rng(a,b):
        i=bisect.bisect_left(tsl,a); j=bisect.bisect_right(tsl,b); return rts_m[i:j]
    def kin(u):
        ks=[]
        for g in rng(u['ts'],u['ts']+u['dur']): ks+=kbyc.get(g['args'].get('correlation'),[])
        return ks
    steps=[]
    for s in rows:
        dk=kin(s['dr']); ek=kin(s['de']); vk_launch=[g for g in rng(s['v']['ts'],s['v']['ts']+s['v']['dur']) if g['name']=='hipGraphLaunch']
        vk=[]
        for g in rng(s['v']['ts'],s['v']['ts']+s['v']['dur']): vk+=kbyc.get(g['args'].get('correlation'),[])
        if not dk or not ek: continue
        d0=min(k['ts'] for k in dk); d1=max(k['ts']+k['dur'] for k in dk)
        e0=min(k['ts'] for k in ek); e1=max(k['ts']+k['dur'] for k in ek)
        vl=vk_launch[0] if vk_launch else None
        # sync preceding verify launch inside run_batch
        sy=[x for x in syn if s['rb']['ts']<=x['ts']<=(vl['ts'] if vl else s['v']['ts']) and x['dur']>500]
        steps.append(dict(bs=int(re.search(r'bs=(\d+)',s['v']['name']).group(1)),d0=d0,d1=d1,e0=e0,e1=e1,
            vl0=vl['ts'] if vl else None,vl1=(vl['ts']+vl['dur']) if vl else None,nvk=len(vk),
            sync_end=(sy[-1]['ts']+sy[-1]['dur']) if sy else None, rb0=s['rb']['ts'],
            dbusy=sum(k['dur'] for k in dk),ebusy=sum(k['dur'] for k in ek)))
    allr[rank]=steps
    n=len(steps)
    per=[ (steps[i+1]['d0']-steps[i]['d0'])/1e3 for i in range(n-1)]
    dspan=[(s['d1']-s['d0'])/1e3 for s in steps]; espan=[(s['e1']-s['e0'])/1e3 for s in steps]
    vwin=[(s['e0']-s['d1'])/1e3 for s in steps]
    lb=[max(0,(s['vl0']-s['d1'])/1e3) for s in steps if s['vl0']]
    lb1=[max(0,(s['vl1']-s['d1'])/1e3) for s in steps if s['vl1']]
    syncl=[(s['sync_end']-s['d1'])/1e3 for s in steps if s['sync_end']]
    gap_prev=[(steps[i+1]['d0']-steps[i]['e1'])/1e3 for i in range(n-1)]
    late=[(steps[i+1]['rb0']-steps[i]['e1'])/1e3 for i in range(n-1)]
    w('\n== rank %s: steps %d, verify-graph kernels recorded total %d'%(rank,n,sum(s['nvk'] for s in steps)))
    w(' bs (verify annot) counts',dict(collections.Counter(s['bs'] for s in steps)),' mean %.2f'%S.mean(s['bs'] for s in steps))
    w(' step period (draft start->next draft start) ms p10/p50/p90/max: %.2f %.2f %.2f %.2f mean %.2f'%(pct(per,10),pct(per,50),pct(per,90),max(per),S.mean(per)))
    w(' draft (4 MTP fwd) GPU span ms p50/p90: %.3f %.3f ; kernel-sum mean %.3f'%(pct(dspan,50),pct(dspan,90),S.mean(s['dbusy'] for s in steps)/1e3))
    w(' draft_extend GPU span ms p50/p90: %.3f %.3f ; kernel-sum mean %.3f'%(pct(espan,50),pct(espan,90),S.mean(s['ebusy'] for s in steps)/1e3))
    w(' verify window (draft end -> draft_extend start) ms p10/p50/p90: %.2f %.2f %.2f mean %.2f'%(pct(vwin,10),pct(vwin,50),pct(vwin,90),S.mean(vwin)))
    w(' CPU blocking D2H sync (graph_variants select) returns after draft GPU end by ms p50/p90: %.3f %.3f'%(pct(syncl,50),pct(syncl,90)))
    w(' verify hipGraphLaunch start - draft GPU end (lower bound of GPU idle before verify) ms p10/p50/p90/mean: %.3f %.3f %.3f %.3f'%(pct(lb,10),pct(lb,50),pct(lb,90),S.mean(lb)))
    w(' verify hipGraphLaunch end - draft GPU end ms p50/p90/mean: %.3f %.3f %.3f'%(pct(lb1,50),pct(lb1,90),S.mean(lb1)))
    w(' GPU gap draft_extend end -> next draft start ms p50/p90/max/mean: %.3f %.3f %.3f %.3f'%(pct(gap_prev,50),pct(gap_prev,90),max(gap_prev),S.mean(gap_prev)))
    w(' next CPU run_batch start - draft_extend GPU end ms (neg = CPU ahead) p50/p90/max: %.2f %.2f %.2f'%(pct(late,50),pct(late,90),max(late)))
    w(' step share of mean period: draft %.1f%% draft_extend %.1f%% verify-window %.1f%% gap_between %.1f%%'%(100*S.mean(dspan)/S.mean(per),100*S.mean(espan)/S.mean(per),100*S.mean(vwin)/S.mean(per),100*S.mean(gap_prev)/S.mean(per)))
# cross-rank alignment: match steps by draft start nearest to rank0
w('\n== cross-rank (rank0 reference; matched by nearest draft start)')
r0=allr['0']
for rk,st in sorted(allr.items()):
    off=[];dend=[];eend=[]
    for s in r0:
        m=min(st,key=lambda x:abs(x['d0']-s['d0']))
        if abs(m['d0']-s['d0'])>20000: continue
        off.append((m['d0']-s['d0'])/1e3); dend.append((m['d1']-s['d1'])/1e3); eend.append((m['e0']-s['e0'])/1e3)
    w(' rank %s: n %d draft-start offset ms p50 %.3f [min %.3f max %.3f]; draft-end offset p50 %.3f; draft_extend-start (=verify end) offset p50 %.3f [min %.3f max %.3f]'%(rk,len(off),pct(off,50),min(off),max(off),pct(dend,50),pct(eend,50),min(eend),max(eend)))
# per-step verify window across ranks
w('\n== per matched step: bs per rank, verify window per rank (ms)')
for s in r0:
    line=[];ok=True
    for rk in sorted(allr):
        m=min(allr[rk],key=lambda x:abs(x['d0']-s['d0']))
        if abs(m['d0']-s['d0'])>20000: ok=False;break
        line.append('%s:bs%d v%.2f lb%.2f'%(rk,m['bs'],(m['e0']-m['d1'])/1e3,max(0,(m['vl0']-m['d1'])/1e3)))
    if ok: w('  '+'  '.join(line))
json.dump(allr,open(outp+'.json','w'))
