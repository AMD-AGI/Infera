import gzip, json, sys, collections
path, steps = sys.argv[1], int(sys.argv[2])
ev = json.load(gzip.open(path, "rt"))["traceEvents"]
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
span = (min(e["ts"] for e in k), max(e["ts"] + e["dur"] for e in k))
tot = collections.defaultdict(lambda: [0.0, 0])
for e in k:
    t = tot[e["name"]]; t[0] += e["dur"]; t[1] += 1
# busy time (union of all kernel intervals) per step
iv = sorted((e["ts"], e["ts"] + e["dur"]) for e in k)
busy, cur = 0.0, None
for a, b in iv:
    if cur is None or a > cur[1]:
        if cur: busy += cur[1] - cur[0]
        cur = [a, b]
    else: cur[1] = max(cur[1], b)
busy += cur[1] - cur[0]
ksum = sum(v[0] for v in tot.values())
streams = collections.Counter(e["args"].get("stream", e.get("tid")) for e in k)
eng = [e for e in k if "engram" in e["name"]]
def overlap(e):
    a, b = e["ts"], e["ts"] + e["dur"]; s = e["args"].get("stream", e.get("tid")); segs = []
    for o in k:
        if o is e or o["args"].get("stream", o.get("tid")) == s: continue
        lo, hi = max(a, o["ts"]), min(b, o["ts"] + o["dur"])
        if lo < hi: segs.append((lo, hi))
    segs.sort(); cov, c = 0.0, None
    for lo, hi in segs:
        if c is None or lo > c[1]:
            if c: cov += c[1] - c[0]
            c = [lo, hi]
        else: c[1] = max(c[1], hi)
    if c: cov += c[1] - c[0]
    return cov / e["dur"] if e["dur"] else 0.0
out = {"trace": path, "steps": steps, "window_span_ms_per_step": (span[1]-span[0])/1000/steps,
       "gpu_busy_union_ms_per_step": busy/1000/steps, "kernel_sum_ms_per_step": ksum/1000/steps,
       "num_streams": len(streams), "kernels_per_stream": dict(streams.most_common()),
       "top": [{"name": n, "ms_per_step": v[0]/1000/steps, "calls_per_step": v[1]/steps,
                "share_of_kernel_sum": v[0]/ksum, "share_of_busy": v[0]/busy} for n, v in sorted(tot.items(), key=lambda x: -x[1][0])[:30]],
       "engram": [{"name": e["name"], "dur_ms": e["dur"]/1000, "stream": e["args"].get("stream", e.get("tid")),
                   "overlapped_by_other_streams_frac": overlap(e)} for e in eng]}
print(json.dumps(out, indent=1))
