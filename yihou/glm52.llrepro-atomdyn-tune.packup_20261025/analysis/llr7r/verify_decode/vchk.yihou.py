# Independent verifier for decode trace (yihou). Usage: python3 -I vchk.yihou.py <trace.json.gz>
import gzip,json,sys,collections,re,statistics as S
d=json.load(gzip.open(sys.argv[1])); ev=d['traceEvents']
print('top-level keys',[k for k in d if k!='traceEvents'])
X=[e for e in ev if e.get('ph')=='X']
cats=collections.Counter(e.get('cat') for e in ev); print('cats',cats.most_common())
phs=collections.Counter(e.get('ph') for e in ev); print('ph',phs)
# GPU-side events = any cat not cpu/python
gpucats={'kernel','gpu_memcpy','gpu_memset','gpu_user_annotation'}
G=sorted([e for e in X if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')],key=lambda e:e['ts'])
print('gpu events',len(G),'pids',collections.Counter((e['pid'],e['tid']) for e in G).most_common(10))
t0=G[0]['ts']; t1=max(e['ts']+e['dur'] for e in G)
# gaps > 5 ms in GPU timeline (union)
gaps=[];ce=G[0]['ts']+G[0]['dur']
for e in G[1:]:
    if e['ts']-ce>5000: gaps.append((e['ts']-ce)/1e3)
    ce=max(ce,e['ts']+e['dur'])
print('span ms %.1f, gpu kernel-sum ms %.1f, #gaps>5ms %d, gap ms p50 %.2f sum %.1f'%((t1-t0)/1e3,sum(e['dur'] for e in G)/1e3,len(gaps),S.median(gaps) if gaps else 0,sum(gaps)))
# any events of any cat in the GPU timeline lanes during a gap? check events with pid matching GPU pid
gpids={e['pid'] for e in G}
others=collections.Counter((e.get('cat'),e.get('name')[:40]) for e in X if e['pid'] in gpids and e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset'))
print('other events on gpu pid',others.most_common(15))
# hipGraphLaunch: correlations with no kernel
corr=collections.Counter(e['args'].get('correlation') for e in G)
rt=[e for e in X if e.get('cat')=='cuda_runtime']
gl=[e for e in rt if e['name']=='hipGraphLaunch']
nk=[corr.get(e['args'].get('correlation'),0) for e in gl]
print('hipGraphLaunch n',len(gl),'kernels/launch counter',collections.Counter(nk).most_common())
print('graphlaunch durs ms', sorted(round(e['dur']/1e3,2) for e in gl if corr.get(e['args'].get('correlation'),0)==0)[:5], '...')
# ac2g flows
fl=[e for e in ev if e.get('ph') in ('s','f')]; print('flow events',len(fl))
# step annotations
ua=[e for e in X if e.get('cat')=='user_annotation']
print('user_annotation names',collections.Counter(re.sub(r'\d+','N',e['name']) for e in ua).most_common(20))
