import gzip,json,sys,collections
d=json.load(gzip.open(sys.argv[1])); ev=d['traceEvents']
X=[e for e in ev if e.get('ph')=='X']
rt={e['args'].get('correlation'):e for e in X if e.get('cat')=='cuda_runtime'}
ks=sorted([e for e in X if e.get('cat') in('kernel','gpu_memcpy')],key=lambda e:e['ts'])
bylaunch=collections.Counter()
for k in ks:
    r=rt.get(k['args'].get('correlation')); bylaunch[r['name'] if r else None]+=1
print(bylaunch)
t0=min(e['ts'] for e in X if e.get('cat')=='kernel'); t1=max(e['ts']+e['dur'] for e in X if e.get('cat')=='kernel')
print('kernel span ms',(t1-t0)/1e3)
ua=sorted([e for e in X if e.get('cat') in('user_annotation','gpu_user_annotation') or (e.get('cat')=='cuda_runtime' and e['name'] in('hipGraphLaunch','hipEventSynchronize','hipDeviceSynchronize','hipMemcpyWithStream'))],key=lambda e:e['ts'])
# print first ~3 steps
st=[e for e in ua if e['name']=='scheduler.run_batch' and e['cat']=='user_annotation']
print('runbatch',len(st))
a=st[3]['ts']; b=st[6]['ts']
for e in ua:
    if a<=e['ts']<b: print('%10.3f %8.3f %s %s %s'%((e['ts']-a)/1e3,e['dur']/1e3,e['cat'],e['tid'],e['name']))
# GPU busy in that window
kk=[k for k in ks if a<=k['ts']<b]
print('nk',len(kk))
prev=None
for k in kk[:0]: pass
