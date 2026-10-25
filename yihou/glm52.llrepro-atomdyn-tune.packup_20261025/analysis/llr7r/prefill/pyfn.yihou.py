import gzip, json, sys, collections, re
f=sys.argv[1]
d=json.load(gzip.open(f))
ev=d['traceEvents']
rx=re.compile(sys.argv[2] if len(sys.argv)>2 else r'draft|eagle|hicache|cache_controller|mooncake|transfer|pack|backup|write_|load_back|_pp_|send|recv|disagg|bootstrap|memcpy|copy_|item|tolist|synchron|cumsum',re.I)
dur=collections.defaultdict(float); cnt=collections.Counter(); tids=collections.defaultdict(set)
for e in ev:
    if e.get('ph')=='X' and e.get('cat') in('python_function','cpu_op') and rx.search(e['name']):
        k=(e['cat'],e['name'][:140]); dur[k]+=e['dur']; cnt[k]+=1; tids[k].add(e['tid'])
for k,v in sorted(dur.items(),key=lambda x:-x[1])[:120]: print('%10.1f %7d %s %s'%(v/1e3,cnt[k],sorted(tids[k])[:4],k))
