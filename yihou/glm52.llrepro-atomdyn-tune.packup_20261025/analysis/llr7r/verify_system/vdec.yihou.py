import sys, gzip, json, bisect, collections
q = lambda a, p: sorted(a)[min(len(a)-1, int(p*len(a)))] if a else float('nan')
f = sys.argv[1]
j = json.load(gzip.open(f)); ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
print("file", f.split('/')[-1])
# thread ids per category
tids = collections.Counter((e.get("cat"), e.get("pid"), e.get("tid")) for e in ev if e.get("cat") in ("python_function","cuda_runtime","user_annotation"))
print("cpu threads:", tids.most_common(8))
rb_cpu = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e["name"]=="scheduler.run_batch" and e.get("cat")=="user_annotation")
rb_gpu = sorted((e["ts"], e["ts"]+e["dur"]) for e in ev if e["name"]=="scheduler.run_batch" and e.get("cat")=="gpu_user_annotation")
print("run_batch cpu n", len(rb_cpu), "dur p50 ms", q([b-a for a,b in rb_cpu],.5)/1e3, " gpu n", len(rb_gpu))
ann = collections.Counter(e["name"] for e in ev if e.get("cat")=="user_annotation")
print("user_annotation names:", ann.most_common(15))
sel = sorted((e["ts"], e["ts"]+e["dur"], e.get("tid")) for e in ev if e.get("cat")=="python_function" and "graph_variants.py(32): select" in e["name"])
print("select n", len(sel), "durs ms sorted:", " ".join(f"{(b-a)/1e3:.1f}" for a,b,_ in sorted(sel, key=lambda t:t[1]-t[0])))
# nesting: is select inside cpu run_batch?
st=[a for a,b in rb_cpu]
ins=0
for a,b,_ in sel:
    i=bisect.bisect_right(st,a)-1
    if i>=0 and rb_cpu[i][1]>=b: ins+=1
print("select inside cpu run_batch:", ins, "/", len(sel))
# memcpy inside select
mc = sorted((e["ts"], e["dur"]) for e in ev if e.get("cat")=="cuda_runtime" and e["name"]=="hipMemcpyWithStream")
big = [(a,b) for a,b,_ in sel if b-a>1000]
for a,b in big[:3]:
    inner=[d for t,d in mc if a<=t<=b]
    print(f" big select {(b-a)/1e3:.1f} ms contains memcpy durs ms", [round(d/1e3,2) for d in inner])
# per step stack: python function names directly enclosing big select (depth)
gl = sorted(e["ts"] for e in ev if e.get("cat")=="cuda_runtime" and e["name"]=="hipGraphLaunch")
print("hipGraphLaunch n", len(gl))
# graph launches per cpu run_batch
per=[sum(1 for t in gl if a<=t<=b) for a,b in rb_cpu]
print("graphlaunch per run_batch:", collections.Counter(per))
# position of graph launches vs big select within each run_batch
for a,b in rb_cpu[5:8]:
    gs=[round((t-a)/1e3,2) for t in gl if a<=t<=b]
    ss=[(round((x-a)/1e3,2),round((y-a)/1e3,2)) for x,y,_ in sel if a<=x<=b]
    print(f" run_batch cpu [{0},{(b-a)/1e3:.1f}] ms graphlaunch@{gs} select@{ss}")
# gpu annotation vs cpu annotation of steps
for (a,b),(c,d) in list(zip(rb_cpu,rb_gpu))[5:8]:
    print(f" cpu rb {a:.0f}-{b:.0f} gpu rb {c:.0f}-{d:.0f} (gpu start-cpu start {(c-a)/1e3:.2f} ms, gpu end - cpu end {(d-b)/1e3:.2f} ms)")
gaps=[rb_gpu[i+1][0]-rb_gpu[i][1] for i in range(len(rb_gpu)-1)]
print("gpu rb gaps us:", " ".join(f"{g:.0f}" for g in gaps))
# kernels: any kernel with correlation to hipGraphLaunch?
corr = {e["args"].get("correlation") for e in ev if e.get("cat")=="cuda_runtime" and e["name"]=="hipGraphLaunch" and "args" in e}
kg=[e for e in ev if e.get("cat") in ("kernel","gpu_memcpy","gpu_memset") and e.get("args",{}).get("correlation") in corr]
print("gpu ops correlated to hipGraphLaunch:", len(kg))
ks=[e for e in ev if e.get("cat")=="kernel"]
print("kernel events", len(ks), "sum ms", sum(e["dur"] for e in ks)/1e3)
