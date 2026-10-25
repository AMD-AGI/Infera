# Independent verifier part 2 (yihou): kernels inside verify window, gaps vs syncs, full name list.
import gzip,json,sys,collections,re,statistics as S,bisect
d=json.load(gzip.open(sys.argv[1])); ev=d['traceEvents']
X=[e for e in ev if e.get('ph')=='X']
G=sorted([e for e in X if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
byc=collections.defaultdict(list)
for e in G: byc[e['args'].get('correlation')].append(e)
rt=sorted([e for e in X if e.get('cat')=='cuda_runtime'],key=lambda e:e['ts'])
gl=[e for e in rt if e['name']=='hipGraphLaunch']
# classify graph launches by kernel count order: expect triplets draft(>100), verify(0), draft_extend(~50)
seq=[(e,len(byc.get(e['args'].get('correlation'),[]))) for e in gl]
print('graph launch kernel-count sequence head',[n for _,n in seq[:12]])
steps=[]
for i in range(len(seq)-2):
    if seq[i][1]>100 and seq[i+1][1]==0 and 30<seq[i+2][1]<100:
        dk=byc[seq[i][0]['args']['correlation']]; ek=byc[seq[i+2][0]['args']['correlation']]
        steps.append(dict(d0=min(k['ts'] for k in dk),d1=max(k['ts']+k['dur'] for k in dk),e0=min(k['ts'] for k in ek),e1=max(k['ts']+k['dur'] for k in ek),
          vl0=seq[i+1][0]['ts'],vl1=seq[i+1][0]['ts']+seq[i+1][0]['dur'],dn=len(dk),en=len(ek)))
print('steps',len(steps),'draft kernels/launch',collections.Counter(s['dn'] for s in steps),'de kernels/launch',collections.Counter(s['en'] for s in steps))
ts=[e['ts'] for e in G]
inwin=collections.Counter(); inwin_t=0; pre=collections.Counter()
for s in steps:
    i=bisect.bisect_right(ts,s['d1']); j=bisect.bisect_left(ts,s['e0'])
    for k in G[i:j]: inwin[k['name'][:70]]+=1; inwin_t+=k['dur']
    i2=bisect.bisect_left(ts,s['vl0'])
    for k in G[bisect.bisect_right(ts,s['d1']):i2]: pre[k['name'][:60]]+=1
print('GPU events strictly inside verify windows (all steps): n',sum(inwin.values()),'sum ms %.3f'%(inwin_t/1e3),inwin.most_common(8))
print('events between draft end and verify graph launch:',sum(pre.values()),pre.most_common(5))
per=[(steps[i+1]['d0']-steps[i]['d0'])/1e3 for i in range(len(steps)-1)]
vw=[(s['e0']-s['d1'])/1e3 for s in steps]
print('period p50 %.2f mean %.2f; verify window p50 %.2f mean %.2f; share %.3f'%(S.median(per),S.mean(per),S.median(vw),S.mean(vw),S.mean(vw)/S.mean(per)))
print('draft span p50 %.3f de span p50 %.3f'%(S.median([(s['d1']-s['d0'])/1e3 for s in steps]),S.median([(s['e1']-s['e0'])/1e3 for s in steps])))
lb=[(s['vl0']-s['d1'])/1e3 for s in steps]; print('vl0-d1 p50 %.3f mean %.3f min %.3f'%(S.median(lb),S.mean(lb),min(lb)))
# between-step gaps and syncs
mt=collections.Counter(e['tid'] for e in rt).most_common(1)[0][0]
ua=[e for e in X if e.get('cat')=='user_annotation' and e['tid']==mt]
gn=[u for u in ua if u['name']=='scheduler.get_next_batch_to_run']
syn=[e for e in rt if e['name']=='hipMemcpyWithStream' and e['dur']>1000]
sg=[s for s in syn if any(u['ts']<=s['ts']<=u['ts']+u['dur'] for u in gn)]
print('get_next_batch syncs >1ms:',[(round(s['dur']/1e3,1)) for s in sg])
for i in range(len(steps)-1):
    g=(steps[i+1]['d0']-steps[i]['e1'])/1e3
    if g>1.0:
        hit=[round(s['dur']/1e3,1) for s in sg if steps[i]['d0']-60000<=s['ts']+s['dur']<=steps[i+1]['d0']]
        # sync end relative to de end
        rel=[round((s['ts']+s['dur']-steps[i]['e1'])/1e3,2) for s in sg if abs(s['ts']+s['dur']-steps[i]['e1'])<5000]
        print('  step %d gap %.2f ms ; get_next syncs ending near de end (rel ms) %s'%(i,g,rel))
big=[(steps[i+1]['d0']-steps[i]['e1'])/1e3 for i in range(len(steps)-1)]
print('gap>1ms count',sum(1 for g in big if g>1),'sum of big/step %.3f small/step %.3f'%(sum(g for g in big if g>1)/len(big),sum(g for g in big if g<=1)/len(big)))
# full names in the draft+de launches w/ ms/step
nm=collections.defaultdict(lambda:[0,0.0])
for s in steps: pass
dc=set();
for i in range(len(seq)-2):
    if seq[i][1]>100 and seq[i+1][1]==0: 
        for k in byc[seq[i][0]['args']['correlation']]: nm[('D',k['name'])][0]+=1; nm[('D',k['name'])][1]+=k['dur']
        for k in byc[seq[i+2][0]['args']['correlation']]: nm[('E',k['name'])][0]+=1; nm[('E',k['name'])][1]+=k['dur']
n=len(steps)
print('graph kernel sum per step ms: draft %.3f de %.3f'%(sum(v[1] for k,v in nm.items() if k[0]=='D')/1e3/n,sum(v[1] for k,v in nm.items() if k[0]=='E')/1e3/n))
for (p,k),(c,t) in sorted(nm.items(),key=lambda x:-x[1][1]):
    print('  %s %7.4f %5.1f  %s'%(p,t/1e3/n,c/n,k.replace('void ','')[:150]))
# non-graph kernels per step
ng=collections.defaultdict(float)
gcorr={e['args'].get('correlation') for e in gl}
for e in G:
    if e['args'].get('correlation') not in gcorr: ng[e['name'][:90]]+=e['dur']
print('non-graph kernels total ms/step %.3f'%(sum(ng.values())/1e3/n))
for k,v in sorted(ng.items(),key=lambda x:-x[1])[:10]: print('   %.4f %s'%(v/1e3/n,k))
