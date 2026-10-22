import sys,collections,statistics as st
def load(f):
    d=collections.defaultdict(dict)
    for ln in open(f):
        t,leg,r,m,v=ln.split('\t'); d[round(float(t))][(int(r),m)]=float(v)
    ts=sorted(d); 
    act=[t for t in ts if sum(d[t].get((r,'num_running_reqs'),0) for r in range(4))>0]
    t0,t1=act[0],act[-1]
    rows=[]
    for t in ts:
        if t<t0+600 or t>t1-120: continue   # skip first 10 min of activity, last 2 min
        x=d[t]
        try:
            run=[x[(r,'num_running_reqs')] for r in range(4)]; gen=[x[(r,'gen_throughput')] for r in range(4)]
        except KeyError: continue
        if sum(gen)<=0: continue
        rows.append((t,run,gen))
    return rows
B=[(0,3),(3,5),(5,7),(7,9),(9,12),(12,16),(16,25)]
def b(m):
    for i,(a,c) in enumerate(B):
        if a<=m<c: return i
res={}
for f in sys.argv[1:]:
    rows=load(f); 
    # per-token latency per sample = sum(run)/sum(gen) ms ; weight by tokens (sum gen)
    W=sum(sum(g) for _,_,g in rows); lat=sum(sum(r) for _,r,_ in rows)/W*1000
    tot=st.mean(sum(r) for _,r,_ in rows); mx=st.mean(max(r) for _,r,_ in rows)
    bins=collections.defaultdict(lambda:[0,0])
    for _,r,g in rows:
        k=b(max(r)); bins[k][0]+=sum(r); bins[k][1]+=sum(g)
    res[f]=(rows,bins)
    print(f"{f:16s} n={len(rows)} dur={rows[-1][0]-rows[0][0]}s tokwLat={lat:.2f}ms meanTotRun={tot:.1f} meanMaxRankRun={mx:.2f} | "+" ".join(f"{B[k]}:{bins[k][0]/bins[k][1]*1000:.1f}({bins[k][1]/W*100:.0f}%)" for k in sorted(bins)))
# counterfactual: pooled idxf1 curve by max-rank bin applied to each run's token distribution
pool=collections.defaultdict(lambda:[0,0])
for f,(rows,bins) in res.items():
    if f.startswith('idxf1'):
        for k,v in bins.items(): pool[k][0]+=v[0]; pool[k][1]+=v[1]
for f,(rows,bins) in res.items():
    W=sum(v[1] for v in bins.values()); act=sum(v[0] for v in bins.values())/W*1000
    pred=sum(v[1]/W*pool[k][0]/pool[k][1] for k,v in bins.items())*1000
    print(f"{f:16s} actual {act:.2f} predicted(idxf1 curve, max-rank bins) {pred:.2f} resid {act-pred:+.2f}")
