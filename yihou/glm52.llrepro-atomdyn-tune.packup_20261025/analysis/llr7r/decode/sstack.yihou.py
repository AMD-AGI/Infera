import gzip,json,sys,collections
d=json.load(gzip.open(sys.argv[1])); ev=d['traceEvents']; X=[e for e in ev if e.get('ph')=='X']
pf=[e for e in X if e.get('cat') in('python_function','cpu_op') and e['tid']==299]
seen=set()
for r in sorted([e for e in X if e.get('cat')=='cuda_runtime' and e['name']=='hipMemcpyWithStream' and e['dur']>1000],key=lambda e:-e['dur'])[:60]:
    enc=sorted([p for p in pf if p['ts']<=r['ts'] and p['ts']+p['dur']>=r['ts']+r['dur']],key=lambda p:-p['dur'])
    key=tuple(p['name'] for p in enc[-8:])
    if key in seen: continue
    seen.add(key); print('dur %.1f ms'%(r['dur']/1e3)); [print('   ',p['name'][:150],'%.2f'%(p['dur']/1e3)) for p in enc if 'sglang' in p['name'] or p['cat']=='cpu_op'][-12:]
