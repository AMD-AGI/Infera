import sys, gzip, json, bisect, collections
q = lambda a, p: sorted(a)[min(len(a)-1, int(p*len(a)))] if a else float('nan')
for f in sys.argv[1:]:
    j = json.load(gzip.open(f)); ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
    gls = sorted((e["ts"], e["ts"]+e["dur"], e["args"]["correlation"]) for e in ev if e.get("cat")=="cuda_runtime" and e["name"]=="hipGraphLaunch")
    byc = collections.defaultdict(list)
    for e in ev:
        if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"): byc[e.get("args",{}).get("correlation")].append(e)
    rb = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e["name"]=="scheduler.run_batch" and e.get("cat")=="user_annotation")
    sel = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e.get("cat")=="python_function" and "graph_variants.py(32): select" in e["name"] and e["dur"]>1000)
    ops = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset"))
    os_=[o[0] for o in ops]
    bub=[]; bub2=[]; rows=[]
    for a,b in rb:
        L=[g for g in gls if a<=g[0]<=b]
        if len(L)!=3: continue
        k1=byc.get(L[0][2],[]); k2=byc.get(L[1][2],[])
        if not k1: continue
        e1=max(e["ts"]+e["dur"] for e in k1)
        # any recorded GPU op ending after e1 and starting before verify launch end?
        s=[x for x in sel if a<=x[0]<=b]
        sr = s[0][1] if s else None
        # first recorded GPU op starting after e1
        i=bisect.bisect_right(os_, e1); nxt = ops[i][0] if i<len(ops) else None
        bub.append(L[1][0]-e1); bub2.append(L[1][1]-L[1][0])  # verify launch API return minus draft-graph GPU end: lower bound of GPU idle before verify
        rows.append((round((e1-a)/1e3,2), round((sr-a)/1e3,2) if sr else None, round((L[1][0]-a)/1e3,2), round((L[1][1]-e1),0), len(k2)))
    print("launchAPI dur us p50", round(q(bub2,.5))); print(f.split("/")[-1][-24:], "LB steps", len(rows), "bubble(draft GPU end -> verify hipGraphLaunch return) us p50/p90/min/max", round(q(bub,.5)), round(q(bub,.9)), round(min(bub)), round(max(bub)), "sum ms", round(sum(bub)/1e3,1))
    if False:
        print(" sample rows (draftGPUend ms, select return ms, verify launch ms rel. cpu run_batch start; bubble us; verify ops recorded)")
        for r in rows[:12]: print("  ", r)
