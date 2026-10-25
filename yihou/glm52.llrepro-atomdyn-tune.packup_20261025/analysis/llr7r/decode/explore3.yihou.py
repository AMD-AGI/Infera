import gzip,json,sys,collections
d=json.load(gzip.open(sys.argv[1])); ev=d['traceEvents']
print({k:d[k] for k in ('roctracer_version','hip_runtime_version','distributedInfo','traceName')})
print([ (p.get('id'),p.get('name')) for p in d['deviceProperties']][:10])
M=[e for e in ev if e.get('ph')=='M']; print([ (e['name'],e.get('pid'),e.get('tid'),e.get('args')) for e in M][:70])
X=[e for e in ev if e.get('ph')=='X']
g=sorted([e for e in X if e.get('cat')=='gpu_user_annotation' and e['name']=='scheduler.run_batch'],key=lambda e:e['ts'])
a=g[10]['ts']; b=g[11]['ts']
c=collections.Counter((e.get('cat'),e.get('pid'),e.get('tid')) for e in ev if a<=e.get('ts',0)<b and e.get('pid')!=299)
print(c)
# graph launches: correlations and kernels per graphlaunch
rt={e['args'].get('correlation'):e for e in X if e.get('cat')=='cuda_runtime'}
gl=[e for e in X if e.get('cat')=='cuda_runtime' and e['name']=='hipGraphLaunch']
kc=collections.Counter(e['args'].get('correlation') for e in X if e.get('cat') in('kernel','gpu_memcpy'))
print('graph launches',len(gl)); 
cnts=collections.Counter(kc.get(e['args']['correlation'],0) for e in gl); print('kernels per graphlaunch hist',sorted(cnts.items()))
big=[e for e in gl if e['dur']>500]; print('graphlaunch >0.5ms dur', len(big), [kc.get(e['args']['correlation'],0) for e in big][:10])
i=[e for e in ev if e.get('ph')=='i']; print(i)
