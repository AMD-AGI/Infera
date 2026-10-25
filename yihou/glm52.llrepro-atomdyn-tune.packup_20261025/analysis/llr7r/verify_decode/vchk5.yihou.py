# verifier part 5 (yihou): verify-graph end bound from the first same-stream kernel after it.
import gzip,json,sys,collections,statistics as S,bisect,re
def pct(a,p): a=sorted(a); return a[min(len(a)-1,int(p/100*len(a)))]
for f in sys.argv[1:]:
    d=json.load(gzip.open(f)); ev=d['traceEvents']; X=[e for e in ev if e.get('ph')=='X']
    G=sorted([e for e in X if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
    byc=collections.defaultdict(list)
    for e in G: byc[e['args'].get('correlation')].append(e)
    rt=[e for e in X if e.get('cat')=='cuda_runtime']
    gl=sorted([e for e in rt if e['name']=='hipGraphLaunch'],key=lambda e:e['ts'])
    seq=[(e,len(byc.get(e['args'].get('correlation'),[]))) for e in gl]
    ub=[];post=[];vw=[];per=[];d0s=[]
    for i in range(len(seq)-2):
        if seq[i][1]>100 and seq[i+1][1]==0 and 30<seq[i+2][1]<100:
            dk=byc[seq[i][0]['args']['correlation']]; ek=byc[seq[i+2][0]['args']['correlation']]
            s=dk[0]['tid']; d1=max(k['ts']+k['dur'] for k in dk); e0=min(k['ts'] for k in ek); vl0=seq[i+1][0]['ts']
            nxt=[k for k in G if k['tid']==s and k['ts']>d1 and k['ts']<e0]
            if not nxt: continue
            vend=nxt[0]['ts']
            ub.append((vend-max(vl0,d1))/1e3); post.append((e0-vend)/1e3); vw.append((e0-d1)/1e3); d0s.append(min(k['ts'] for k in dk))
    per=[(d0s[j+1]-d0s[j])/1e3 for j in range(len(d0s)-1)]
    print('%s n=%d verify-graph upper bound (first same-stream post kernel start - launch call start) ms p10/p50/p90/mean %.2f %.2f %.2f %.2f ; post-verify..de-graph ms p50 %.3f ; window(d1->de graph) p50 %.2f ; period mean %.2f ; ub share %.3f'%(re.search(r'TP-\d',f).group(0),len(ub),pct(ub,10),pct(ub,50),pct(ub,90),S.mean(ub),pct(post,50),pct(vw,50),S.mean(per),S.mean(ub)/S.mean(per)))
