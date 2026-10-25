import gzip,json,sys,collections
f=sys.argv[1]
d=json.load(gzip.open(f))
ev=d['traceEvents'] if isinstance(d,dict) else d
print(len(ev), [k for k in d.keys()] if isinstance(d,dict) else '')
cat=collections.Counter((e.get('cat'),e.get('ph')) for e in ev)
print(cat.most_common(30))
# pids/tids
pt=collections.Counter((e.get('pid'),e.get('tid'),e.get('cat')) for e in ev if e.get('ph')=='X')
for k,v in pt.most_common(20): print(k,v)
# cpu op names top
cn=collections.Counter(e['name'] for e in ev if e.get('cat') in ('cpu_op','user_annotation','python_function') )
print([x for x in cn.most_common(60)])
ua=collections.Counter(e['name'] for e in ev if e.get('cat') in ('user_annotation','gpu_user_annotation'))
print('UA',ua.most_common(80))
rt=collections.Counter(e['name'] for e in ev if e.get('cat')=='cuda_runtime')
print('RT',rt.most_common(30))
ks=[e for e in ev if e.get('cat')=='kernel']
print('nkern',len(ks)); 
print(ks[0])
k=collections.Counter(e['name'][:120] for e in ks); print(len(k))
for n,c in k.most_common(80): print(c,n)
