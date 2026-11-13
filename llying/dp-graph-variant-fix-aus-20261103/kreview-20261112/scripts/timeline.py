#!/usr/bin/env python3
"""Per-step timeline of one rank: CPU annotations vs GPU graph windows (first kernel start .. last kernel end)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import kl
d = kl.load(sys.argv[1]); n0 = int(sys.argv[2]) if len(sys.argv) > 2 else 5; n = int(sys.argv[3]) if len(sys.argv) > 3 else 3
L = d["launches"]
t0 = L[0]["cpu_ts"]
ev = []
for a in d["ann"]:
    ev.append((a[1], "CPU", a[0], a[2]))
for l in L:
    ev.append((l["cpu_ts"], "CPU", f"  hipGraphLaunch[{l['kind']} bs={l['bs']}]", l["cpu_dur"]))
    if l["k"]:
        s = l["k"][0][1]; e = l["k"][-1][1] + l["k"][-1][2]
        ev.append((s, "GPU", f"graph {l['kind']} bs={l['bs']} nk={len(l['k'])} sumdur={sum(k[2] for k in l['k'])/1e3:.2f}ms", e - s))
for r in d["rt"]:
    if r[0] != "hipGraphLaunch":
        ev.append((r[1], "CPU", "  rt:" + r[0], r[2]))
for g in d["gann"]:
    ev.append((g[1], "GPUANN", g[0], g[2]))
# eager kernels grouped into bursts
E = d["eager"]; burst = None
for k in E:
    if burst and k[1] - burst[1] < 200:
        burst[1] = k[1] + k[2]; burst[2] += 1; burst[3] += k[2]
    else:
        if burst: ev.append((burst[0], "GPU", f"eager burst n={burst[2]} sumdur={burst[3]:.0f}us", burst[1] - burst[0]))
        burst = [k[1], k[1] + k[2], 1, k[2]]
if burst: ev.append((burst[0], "GPU", f"eager burst n={burst[2]} sumdur={burst[3]:.0f}us", burst[1] - burst[0]))
for m in d["mem"]:
    ev.append((m[1], "GPU", f"{m[0]} bytes={m[3]}", m[2]))
ev.sort()
vs = [l for l in L if l["kind"] == "verify"]
lo = vs[n0]["ann_ts"] - 3000; hi = vs[n0 + n]["ann_ts"]
for ts, w, name, dur in ev:
    if lo <= ts <= hi and (dur >= 30 or w != "GPU" or "graph" in name):
        print(f"{(ts - lo) / 1e3:9.3f} ms  +{dur / 1e3:8.3f} ms  {w:6s} {name}")
