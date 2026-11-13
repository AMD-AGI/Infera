#!/usr/bin/env python3
"""Print the kernel sequence of the N-th verify launch of one rank (index, start offset us, dur us, gap to prev, name)."""
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
import kl
d = kl.load(sys.argv[1]); n = int(sys.argv[2]); lim = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
kind = sys.argv[4] if len(sys.argv) > 4 else "verify"
L = [l for l in d["launches"] if l["kind"] == kind]
l = L[n]
print(f"# {d['pname']} {kind} #{n}/{len(L)} bs={l['bs']} kernels={len(l['k'])} cpu_launch_ts={l['cpu_ts']:.1f} dur={l['cpu_dur']:.1f}")
t0 = l["k"][0][1]; prev = t0
print(f"# first kernel starts {t0 - l['cpu_ts']:.1f} us after hipGraphLaunch call; graph span {(l['k'][-1][1] + l['k'][-1][2] - t0) / 1e3:.3f} ms; sum dur {sum(k[2] for k in l['k']) / 1e3:.3f} ms")
for i, (nid, ts, dur, stream) in enumerate(l["k"][:lim]):
    print(f"{i:5d} {ts - t0:10.1f} {dur:8.2f} {ts - prev:7.2f} s{stream} {kl.short(d['names'][nid], 130)}")
    prev = ts + dur
