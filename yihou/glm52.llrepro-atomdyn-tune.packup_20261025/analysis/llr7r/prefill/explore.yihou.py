import gzip, json, sys, collections
f=sys.argv[1]
d=json.load(gzip.open(f))
ev=d['traceEvents'] if isinstance(d,dict) else d
print('n events',len(ev), 'keys', list(d.keys()) if isinstance(d,dict) else None)
cats=collections.Counter((e.get('cat'),e.get('ph')) for e in ev)
for k,v in cats.most_common(40): print(k,v)
# pids/tids
pt=collections.Counter((e.get('pid'),e.get('tid')) for e in ev if e.get('ph')=='X')
for k,v in pt.most_common(20): print('pidtid',k,v)
# GPU kernels
dur=collections.defaultdict(float); cnt=collections.Counter()
ts=[]
for e in ev:
    if e.get('ph')!='X': continue
    c=e.get('cat','')
    if c in ('kernel','gpu_memcpy','gpu_memset'):
        dur[(c,e['name'][:110])]+=e['dur']; cnt[(c,e['name'][:110])]+=1; ts.append(e['ts'])
print('gpu span', (max(ts)-min(ts))/1e3 if ts else None,'ms')
tot=sum(dur.values()); print('gpu tot ms',tot/1e3)
for k,v in sorted(dur.items(),key=lambda x:-x[1])[:70]: print('%9.1f %6d %s'%(v/1e3,cnt[k],k))
# cpu annotations
ua=collections.defaultdict(float); uc=collections.Counter()
for e in ev:
    if e.get('ph')=='X' and e.get('cat') in ('user_annotation','gpu_user_annotation','python_function'):
        if e.get('cat')=='python_function': continue
        ua[(e['cat'],e['name'][:100])]+=e['dur']; uc[(e['cat'],e['name'][:100])]+=1
for k,v in sorted(ua.items(),key=lambda x:-x[1])[:50]: print('ANN %9.1f %6d %s'%(v/1e3,uc[k],k))
