#!/usr/bin/env python3
"""kdiff.py TRACE_A TRACE_B [N]: per kernel name, ms per VERIFY step (kernels overlapping gpu_user_annotation
step[VERIFY ...] spans), A vs B, sorted by |A-B|; prints top N."""
import collections, gzip, json, re, sys

def per_step(path):
    ev = [e for e in json.load(gzip.open(path, "rt"))["traceEvents"] if e.get("ph") == "X"]
    spans = sorted((e["ts"], e["ts"] + e["dur"]) for e in ev
                   if e.get("cat") == "gpu_user_annotation" and e["name"].startswith("step[VERIFY"))
    tot = collections.Counter()
    for k in ev:
        if k.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        mid = k["ts"] + k["dur"] / 2
        if any(s <= mid <= e for s, e in spans):
            tot[re.sub(r"\s+", " ", k["name"])[:90]] += k["dur"] / 1e3
    return {n: v / len(spans) for n, v in tot.items()}, len(spans)

a, na = per_step(sys.argv[1]); b, nb = per_step(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 40
print(f"verify steps A={na} B={nb}; ms/step A, B, A-B")
for k in sorted(set(a) | set(b), key=lambda k: -abs(a.get(k, 0) - b.get(k, 0)))[:n]:
    print(f"{a.get(k, 0):8.3f} {b.get(k, 0):8.3f} {a.get(k, 0) - b.get(k, 0):+8.3f}  {k}")
