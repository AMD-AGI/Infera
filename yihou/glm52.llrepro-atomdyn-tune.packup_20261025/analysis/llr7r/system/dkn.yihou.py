import sys, gzip, json, collections
f = sys.argv[1]; j = json.load(gzip.open(f)); ev = [e for e in j["traceEvents"] if e.get("ph") == "X"]
rt = collections.Counter(); rtd = collections.Counter()
for e in ev:
    if e.get("cat") == "cuda_runtime": rt[e["name"]] += 1; rtd[e["name"]] += e["dur"]
for n, c in rt.most_common(15): print(f"rt {n} n={c} sum={rtd[n]/1e3:.1f} ms")
kn = collections.Counter(); kd = collections.Counter()
for e in ev:
    if e.get("cat") == "kernel": kn[e["name"][:70]] += 1; kd[e["name"][:70]] += e["dur"]
for n, d in kd.most_common(12): print(f"k {d/1e3:8.1f} ms n={kn[n]} {n}")
# run_batch annotations (cpu and gpu) durations
ra = sorted(e["dur"]/1e3 for e in ev if e["name"] == "scheduler.run_batch" and e.get("cat") == "gpu_user_annotation")
print("gpu run_batch ms:", " ".join(f"{x:.1f}" for x in ra))
# per-step: time from select() start to end and the step period
sel = sorted((e["ts"], e["dur"]) for e in ev if e.get("cat") == "python_function" and "graph_variants.py(32): select" in e["name"])
print("select ms:", " ".join(f"{d/1e3:.1f}" for _, d in sel))
