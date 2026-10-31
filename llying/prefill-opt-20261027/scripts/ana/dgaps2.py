"""Largest GPU idle gaps in one decode step window with neighbouring kernels and the CPU annotations/ops running
during the gap (user_annotation + cuda_runtime sync calls). python3 dgaps2.py TRACE.json.gz"""
import gzip, json, sys, re, collections
d = json.load(gzip.open(sys.argv[1])); ev = [e for e in d["traceEvents"] if e.get("ph") == "X"]
k = sorted((e for e in ev if e.get("cat") == "kernel"), key=lambda e: e["ts"])
cpu = [e for e in ev if e.get("cat") in ("cuda_runtime", "user_annotation", "python_function", "cpu_op")]
gaps = []
end, prev = k[0]["ts"] + k[0]["dur"], k[0]
for e in k[1:]:
    if e["ts"] - end > 50: gaps.append((e["ts"] - end, end, e["ts"], prev["name"], e["name"]))
    if e["ts"] + e["dur"] >= end: end, prev = e["ts"] + e["dur"], e
nsteps = sum(1 for e in ev if e.get("name") == "hipGraphLaunch") // 3
# group gaps by (prev kernel, next kernel) signature
sig = collections.defaultdict(list)
s = lambda n: re.sub(r"<.*|\(.*", "", n)[:50]
for g in gaps: sig[(s(g[3]), s(g[4]))].append(g)
for (a, b), v in sorted(sig.items(), key=lambda x: -sum(g[0] for g in x[1]))[:8]:
    tot = sum(g[0] for g in v) / 1e3 / nsteps
    g = max(v)
    # cpu events overlapping the largest such gap: syncs and the innermost python functions
    ov = [c for c in cpu if c["ts"] < g[2] and c["ts"] + c["dur"] > g[1]]
    rt = collections.Counter(c["name"] for c in ov if c["cat"] == "cuda_runtime")
    py = sorted((c for c in ov if c["cat"] == "python_function" and c["dur"] < 5 * (g[2] - g[1]) + 2000), key=lambda c: -min(c["ts"] + c["dur"], g[2]) + max(c["ts"], g[1]))[:6]
    print(f"{tot:.2f} ms/step  n={len(v)}  [{a}] -> [{b}]  largest {g[0]:.0f} us")
    print("     runtime:", dict(rt.most_common(5)))
    for c in py: print(f"     py {min(c['ts'] + c['dur'], g[2]) - max(c['ts'], g[1]):7.0f}us  {c['name'][:110]}")
