import sys, gzip, json, bisect, collections
q = lambda a, p: sorted(a)[min(len(a)-1, int(p*len(a)))] if a else float('nan')
f = sys.argv[1]
j = json.load(gzip.open(f)); ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
gls = sorted((e["ts"], e["args"]["correlation"]) for e in ev if e.get("cat")=="cuda_runtime" and e["name"]=="hipGraphLaunch")
byc = collections.defaultdict(list)
for e in ev:
    if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"):
        byc[e.get("args",{}).get("correlation")].append(e)
print("per graph launch: idx, n_ops, sum_dur ms, first start - launch ms, extent ms, top kernels")
for i,(t,c) in enumerate(gls[:9]):
    k = byc.get(c, [])
    if not k: print(i, 0); continue
    s=min(e["ts"] for e in k); E=max(e["ts"]+e["dur"] for e in k)
    top=collections.Counter()
    for e in k: top[e["name"][:50]]+=e["dur"]
    print(i, len(k), round(sum(e['dur'] for e in k)/1e3,2), round((s-t)/1e3,2), round((E-s)/1e3,2), [(n,round(d/1e3,2)) for n,d in top.most_common(3)])
# all GPU ops union busy, and union of graph extents
ops = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"))
U=[]
for a,b in ops:
    if U and a<=U[-1][1]: U[-1][1]=max(U[-1][1],b)
    else: U.append([a,b])
span=U[-1][1]-U[0][0]; busy=sum(b-a for a,b in U)
print(f"op union busy {busy/1e3:.1f} ms of {span/1e3:.1f} ms = {busy/span:.3f}")
# streams
st=collections.Counter((e.get("tid")) for e in ev if e.get("cat")=="kernel"); print("kernel tids", st.most_common(6))
# kernel duration distribution
kd=[e["dur"] for e in ev if e.get("cat")=="kernel"]; print("kernel dur us p50/p90/max", q(kd,.5), q(kd,.9), max(kd))
# within one run_batch gpu annotation, union busy of kernels
rb = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e["name"]=="scheduler.run_batch" and e.get("cat")=="gpu_user_annotation")
fr=[]
for a,b in rb:
    i=bisect.bisect_left([u[0] for u in U],a)
    bs=sum(max(0,min(y,b)-max(x,a)) for x,y in U if y>a and x<b)
    fr.append(bs/(b-a))
print("kernel-union busy fraction within gpu run_batch annotation p50/min/max", round(q(fr,.5),3), round(min(fr),3), round(max(fr),3))
# show largest idle gaps inside one annotation
a,b=rb[10]
inside=[(x,y) for x,y in U if y>a and x<b]
g=sorted(((inside[i+1][0]-inside[i][1]), round((inside[i][1]-a)/1e3,2)) for i in range(len(inside)-1))[-6:]
print("largest gaps in run_batch #10 (us, at ms):", g)
