#!/usr/bin/env python3
"""How much of a verify graph is kernels at the dispatch floor: count and total time of kernels shorter than 6 us,
by class, plus the distribution of the shortest kernels (the per-kernel floor of HIP graph replay on this stack)."""
import collections, sys, statistics as st
sys.path.insert(0, __import__("os").path.dirname(__file__))
import kl, ana2
files = kl.rank_files(sys.argv[1]); thr = 6.0
small = collections.Counter(); small_t = collections.Counter(); tot_t = 0.0; tot_n = 0; nsteps = 0; alld = []
for f in files[:4]:
    d = kl.load(f)
    for l in d["launches"]:
        if l["kind"] != "verify" or not l["k"]:
            continue
        cls, _ = ana2.parse_verify(d["names"], l["k"])
        nsteps += 1
        for (nid, ts, dur, _s), c in zip(l["k"], cls):
            tot_t += dur; tot_n += 1; alld.append(dur)
            if dur < thr:
                small[c] += 1; small_t[c] += dur
alld.sort()
print(f"{sys.argv[1].split('/')[-1]}: {tot_n / nsteps:.0f} kernels, {tot_t / nsteps / 1e3:.2f} ms per verify; shortest kernels us: p0.1 {alld[len(alld) // 1000]:.2f} p1 {alld[len(alld) // 100]:.2f} p5 {alld[len(alld) // 20]:.2f} p10 {alld[len(alld) // 10]:.2f} p25 {alld[len(alld) // 4]:.2f} p50 {alld[len(alld) // 2]:.2f}")
n = sum(small.values()) / nsteps; t = sum(small_t.values()) / nsteps / 1e3
print(f"  kernels < {thr:.0f} us: {n:.0f} per step ({100 * n * nsteps / tot_n:.0f}% of kernels), {t:.2f} ms ({100 * t * 1e3 * nsteps / tot_t:.0f}% of verify time)")
print("   " + ", ".join(f"{c} {small[c] / nsteps:.0f}x{small_t[c] / small[c]:.1f}us" for c, _ in small.most_common(14)))
print(f"  if every kernel carried a 4.0 us floor: {tot_n / nsteps * 4.0 / 1e3:.2f} ms per verify ({100 * tot_n * 4.0 / tot_t:.0f}%)")
