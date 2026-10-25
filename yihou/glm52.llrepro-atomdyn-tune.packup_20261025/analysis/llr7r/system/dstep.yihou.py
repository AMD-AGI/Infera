import sys, gzip, json, bisect
q = lambda a, p: sorted(a)[min(len(a)-1, int(p*len(a)))]
for f in sys.argv[1:]:
    j = json.load(gzip.open(f)); ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
    g = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e["name"] == "scheduler.run_batch" and e.get("cat") == "gpu_user_annotation")
    gaps = [g[i+1][0]-g[i][1] for i in range(len(g)-1)]; per = [g[i+1][0]-g[i][0] for i in range(len(g)-1)]
    sel = sorted((e["ts"]+e["dur"], e["dur"]) for e in ev if e.get("cat") == "python_function" and "graph_variants.py(32): select" in e["name"] and e["dur"] > 1000)
    gl = sorted(e["ts"] for e in ev if e.get("cat") == "cuda_runtime" and e["name"] == "hipGraphLaunch")
    lag = []
    for r, d in sel:
        i = bisect.bisect_left(gl, r)
        if i < len(gl): lag.append(gl[i]-r)
    print(f"{f[-24:]}: steps={len(g)} GPU run_batch span p50={q([b-a for a,b in g],.5)/1e3:.1f} ms; period p50={q(per,.5)/1e3:.1f} ms; "
          f"GPU gap between run_batch annotations p50={q(gaps,.5):.0f} us p90={q(gaps,.9):.0f} us max={max(gaps):.0f} us sum={sum(gaps)/1e3:.1f} ms ({sum(gaps)/(g[-1][1]-g[0][0]):.3f} of span); "
          f"blocking select() n={len(sel)} p50={q([d for _,d in sel],.5)/1e3:.1f} ms; select-return -> next hipGraphLaunch p50={q(lag,.5)/1e3:.2f} ms p90={q(lag,.9)/1e3:.2f} ms")
