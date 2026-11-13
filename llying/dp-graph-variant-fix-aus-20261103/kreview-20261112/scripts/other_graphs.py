#!/usr/bin/env python3
"""Top kernels of the draft / draft_extend graphs and of the eager (non-graph) kernels, per step, for one capture."""
import collections, sys, statistics as st
sys.path.insert(0, __import__("os").path.dirname(__file__))
import kl
files = kl.rank_files(sys.argv[1]); top = int(sys.argv[2]) if len(sys.argv) > 2 else 12
acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0.0, 0])); nl = collections.Counter(); nsteps = 0
for f in files:
    d = kl.load(f)
    V = [l for l in d["launches"] if l["kind"] == "verify" and l["k"]]
    nsteps += len(V)
    for l in d["launches"]:
        if l["kind"] in ("draft", "draft_extend") and l["k"]:
            nl[l["kind"]] += 1
            for nid, ts, dur, _ in l["k"]:
                a = acc[l["kind"]][kl.short(d["names"][nid], 95)]; a[0] += dur; a[1] += 1
    lo, hi = V[0]["k"][0][1], V[-1]["k"][0][1]
    for k in d["eager"]:
        if lo <= k[1] <= hi:
            a = acc["eager"][kl.short(d["names"][k[0]], 95)]; a[0] += k[2]; a[1] += 1
    for m in d["mem"]:
        if lo <= m[1] <= hi:
            a = acc["memcpy/memset"][m[0]]; a[0] += m[2]; a[1] += 1
for kind in ("draft", "draft_extend", "eager", "memcpy/memset"):
    tot = sum(v[0] for v in acc[kind].values()); n = sum(v[1] for v in acc[kind].values())
    print(f"## {kind}: {tot / nsteps / 1e3:.3f} ms/step, {n / nsteps:.1f} kernels/step")
    for name, (us, c) in sorted(acc[kind].items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"   {us / nsteps:8.1f} us/step {c / nsteps:5.1f} calls {us / c:8.2f} us/call  {name}")
