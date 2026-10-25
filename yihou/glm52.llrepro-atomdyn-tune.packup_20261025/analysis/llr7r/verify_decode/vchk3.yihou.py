# verifier part 3 (yihou): post-verify kernels inside verify window; union idle per step.
import gzip,json,sys,collections,statistics as S,bisect
d=json.load(gzip.open(sys.argv[1])); ev=d['traceEvents']; X=[e for e in ev if e.get('ph')=='X']
G=sorted([e for e in X if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
byc=collections.defaultdict(list)
for e in G: byc[e['args'].get('correlation')].append(e)
rt=[e for e in X if e.get('cat')=='cuda_runtime']; rtc={e['args'].get('correlation'):e for e in rt}
gl=sorted([e for e in rt if e['name']=='hipGraphLaunch'],key=lambda e:e['ts'])
seq=[(e,len(byc.get(e['args'].get('correlation'),[]))) for e in gl]
ts=[e['ts'] for e in G]
fv=[];span=[];lastend=[];tids=collections.Counter();gtid=collections.Counter();firstn=collections.Counter();launchlag=[]
st=[]
for i in range(len(seq)-2):
    if seq[i][1]>100 and seq[i+1][1]==0 and 30<seq[i+2][1]<100:
        dk=byc[seq[i][0]['args']['correlation']]; ek=byc[seq[i+2][0]['args']['correlation']]
        d0=min(k['ts'] for k in dk); d1=max(k['ts']+k['dur'] for k in dk); e0=min(k['ts'] for k in ek); e1=max(k['ts']+k['dur'] for k in ek)
        for k in dk: gtid[k['tid']]+=1
        w=G[bisect.bisect_right(ts,d1):bisect.bisect_left(ts,e0)]
        st.append((d0,d1,e0,e1,seq[i+1][0]))
        if w:
            fv.append((w[0]['ts']-d1)/1e3); span.append((w[-1]['ts']+w[-1]['dur']-w[0]['ts'])/1e3); lastend.append((e0-(w[-1]['ts']+w[-1]['dur']))/1e3)
            for k in w: tids[k['tid']]+=1
            firstn[w[0]['name'][:60]]+=1
            r=rtc.get(w[0]['args'].get('correlation')); 
            if r: launchlag.append((w[0]['ts']-(r['ts']))/1e3)
print('draft graph kernel tids',gtid,'window kernel tids',tids)
print('first event in verify window after draft end ms p10/p50/p90/min: %.2f %.2f %.2f %.2f'%(sorted(fv)[len(fv)//10],S.median(fv),sorted(fv)[9*len(fv)//10],min(fv)))
print('first window event names',firstn.most_common(3))
print('first window event GPU start - its CPU launch ms p50 %.2f min %.2f'%(S.median(launchlag),min(launchlag)))
print('post-verify kernel cluster span ms p50 %.3f ; gap cluster end -> draft_extend start p50 %.3f'%(S.median(span),S.median(lastend)))
# union idle within each period [d0_i, d0_{i+1}) assuming verify graph busy from vl0 to first window event
idle=[]
for j in range(len(st)-1):
    a=st[j][0]; b=st[j+1][0]
    w=G[bisect.bisect_left(ts,a):bisect.bisect_left(ts,b)]
    iv=sorted((k['ts'],k['ts']+k['dur']) for k in w)
    tot=0;cs,ce=iv[0]
    for x,y in iv[1:]:
        if x<=ce: ce=max(ce,y)
        else: tot+=ce-cs; cs,ce=x,y
    tot+=ce-cs
    idle.append(((b-a)-tot)/1e3)
print('recorded-kernel union busy per period: idle (incl. verify graph) mean %.2f ms'%S.mean(idle))
