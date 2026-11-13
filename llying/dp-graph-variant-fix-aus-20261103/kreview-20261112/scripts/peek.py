#!/usr/bin/env python3
"""Dump the raw structure of one torch-profiler trace: event classes, arg keys, annotations, overlap of kernel events."""
import collections, gzip, json, sys
p = sys.argv[1]
d = json.load(gzip.open(p))
print("top keys:", {k: (type(v).__name__, len(v) if hasattr(v, "__len__") else v) for k, v in d.items() if k != "traceEvents"})
ev = d["traceEvents"]
print("events:", len(ev))
c = collections.Counter((e.get("ph"), e.get("cat")) for e in ev)
for k, v in c.most_common():
    print("  ", k, v)
seen = set()
for e in ev:
    k = (e.get("ph"), e.get("cat"))
    if k not in seen:
        seen.add(k)
        print("SAMPLE", k, json.dumps(e)[:900])
ann = collections.Counter(e["name"] for e in ev if e.get("cat") == "user_annotation")
print("user_annotation names:", len(ann))
for k, v in ann.most_common(60):
    print("  ", v, k)
gpu_ann = collections.Counter(e["name"] for e in ev if e.get("cat") == "gpu_user_annotation")
print("gpu_user_annotation:", gpu_ann.most_common(20))
ks = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
print("kernels:", len(ks), "span ms:", (ks[-1]["ts"] + ks[-1]["dur"] - ks[0]["ts"]) / 1e3)
print("kernel arg keys:", collections.Counter(k for e in ks for k in e.get("args", {})).most_common())
print("kernel pid/tid:", collections.Counter((e.get("pid"), e.get("tid")) for e in ks).most_common(10))
for key in ("device", "stream", "context", "queued", "resource"):
    print(" ", key, collections.Counter(e["args"].get(key) for e in ks).most_common(8))
# overlap / gaps between consecutive kernels
ov = 0; ovt = 0.0; gaps = []
for a, b in zip(ks, ks[1:]):
    g = b["ts"] - (a["ts"] + a["dur"])
    gaps.append(g)
    if g < 0:
        ov += 1; ovt += -g
gaps.sort()
print("consecutive kernels overlapping:", ov, "total overlap us:", ovt)
print("gap us pcts:", [gaps[int(len(gaps) * q)] for q in (0.01, 0.1, 0.5, 0.9, 0.99, 0.999)])
print("sum kernel dur ms:", sum(e["dur"] for e in ks) / 1e3)
durs = sorted(e["dur"] for e in ks)
print("dur us pcts:", [durs[int(len(durs) * q)] for q in (0.0, 0.01, 0.1, 0.5, 0.9, 0.99)], "max", durs[-1])
print("ts precision sample:", [(e["ts"], e["dur"]) for e in ks[1000:1006]])
# runtime events
rt = collections.Counter(e["name"] for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver"))
print("runtime:", rt.most_common(30))
gl = [e for e in ev if e.get("name") == "hipGraphLaunch"]
print("hipGraphLaunch:", len(gl), json.dumps(gl[5])[:600] if len(gl) > 5 else "")
corr = collections.Counter(e["args"].get("correlation") for e in ks)
print("distinct correlations among kernels:", len(corr), "top:", corr.most_common(5))
glc = set(e["args"]["correlation"] for e in gl)
print("kernels whose correlation is a graph launch:", sum(1 for e in ks if e["args"].get("correlation") in glc), "/", len(ks))
names = collections.Counter(e["name"] for e in ks)
print("distinct kernel names:", len(names))
for n, v in names.most_common(80):
    print("  ", v, n[:200])
