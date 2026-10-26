import re,sys,collections
pat=re.compile(r"shape is M:(\d+), N:(\d+), K:(\d+) dtype='([^']+)' otype='([^']+)' bias=(\w+), scaleAB=(\w+), bpreshuffle=(\w+), not found tuned config.*using (\w+) solution")
for fn in sys.argv[1:]:
    d=collections.defaultdict(set); lib=collections.defaultdict(collections.Counter); tot=0
    for line in open(fn,errors="ignore"):
        m=pat.search(line)
        if not m: continue
        tot+=1
        M,N,K=map(int,m.group(1,2,3)); key=(N,K,m.group(5),m.group(8))
        d[key].add(M); lib[key][m.group(9)]+=1
    print("==",fn,"lines",tot)
    for k in sorted(d): print(k,dict(lib[k]),sorted(d[k]))
