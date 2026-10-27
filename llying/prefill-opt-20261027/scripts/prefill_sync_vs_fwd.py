import bisect, collections, gzip, json, re, sys
ev = json.load(gzip.open(sys.argv[1], "rt"))["traceEvents"]
gpu = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
st = collections.Counter(k["args"].get("stream") for k in gpu if "nccl" not in k["name"].lower())
main = st.most_common(1)[0][0]
ops = sorted((k for k in gpu if k["args"].get("stream") == main), key=lambda k: k["ts"])
ends = [min(k["ts"] + k["dur"], ops[i + 1]["ts"]) if i + 1 < len(ops) else k["ts"] + k["dur"] for i, k in enumerate(ops)]
# main-stream busy segments separated by idle > 5 ms = one forward each
segs = []
s0, e0 = ops[0]["ts"], ends[0]
for k, e in zip(ops[1:], ends[1:]):
    if k["ts"] - e0 > 5e3:
        segs.append((s0, e0)); s0 = k["ts"]
    e0 = max(e0, e)
segs.append((s0, e0))
seg_s = [s for s, _ in segs]
steps = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[")]
tid = steps[0]["tid"]
side = sorted((k for k in gpu if k["args"].get("stream") != main and "transfer" in k["name"]), key=lambda k: k["ts"])
rows = []
for e in ev:
    if e.get("ph") == "X" and e.get("cat") == "cuda_runtime" and e.get("tid") == tid and e["name"] == "hipEventSynchronize" and e["dur"] > 5e3:
        s, t = e["ts"], e["ts"] + e["dur"]
        i = bisect.bisect_right(seg_s, s) - 1
        fwd_end = segs[i][1] if i >= 0 and segs[i][1] > s else None
        # backup kernels (device_to_host) running between forward end and sync end
        rows.append((e["dur"] / 1e3, (fwd_end - s) / 1e3 if fwd_end else None, (t - fwd_end) / 1e3 if fwd_end else None))
n = len(rows)
inf = [r for r in rows if r[1] is not None]
print(f"syncs > 5 ms: {n}, total {sum(r[0] for r in rows):.0f} ms; {len(inf)} start while a forward is running")
print(f"  of their time: waiting for the running forward to end {sum(min(r[1], r[0]) for r in inf):.0f} ms, after forward end {sum(max(r[2], 0) for r in inf):.0f} ms")
idle = [r for r in rows if r[1] is None]
print(f"  syncs starting with the GPU main stream idle: {len(idle)}, total {sum(r[0] for r in idle):.0f} ms")
for r in sorted(rows, reverse=True)[:12]:
    print("   sync %.1f ms; forward ends %s ms after sync start; sync ends %s ms after forward end" % (r[0], "%.1f" % r[1] if r[1] is not None else "-", "%.1f" % r[2] if r[2] is not None else "-"))
