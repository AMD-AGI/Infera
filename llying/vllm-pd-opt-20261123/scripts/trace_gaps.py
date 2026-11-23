#!/usr/bin/env python3
"""GPU idle gaps in one torch-profiler trace of a decode rank, and the kernels around them.
  trace_gaps.py TRACE.json[.gz] [--step-kernel REGEX] [--min-gap-us 20] [--top 25]
Kernel intervals (all GPU streams) are merged; every gap between merged busy intervals is attributed to the pair
(last kernel before -> first kernel after), names shortened. Steps are counted by --step-kernel (default: the sampler's
top-k / top-p kernel, once per decode step). Prints busy / idle per step and the gap pairs that hold the idle time.
"""
import argparse
import collections
import gzip
import json
import re


def short(name):
    name = re.sub(r"\(.*", "", name)
    name = re.sub(r"<.*", "", name)
    for p in ("void ", "_ZN5aiter", "_ZN2ck"):
        name = name.replace(p, "")
    return name[:60]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--step-kernel", default=r"_topk_topp_kernel")
    ap.add_argument("--min-gap-us", type=float, default=20.0)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    op = gzip.open if a.trace.endswith(".gz") else open
    with op(a.trace, "rt") as f:
        ev = json.load(f)
    ev = ev["traceEvents"] if isinstance(ev, dict) else ev
    ks = sorted((e["ts"], e["ts"] + e.get("dur", 0), e["name"]) for e in ev
                if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"))
    del ev
    if not ks:
        print("no kernels")
        return
    steps = sum(1 for k in ks if re.search(a.step_kernel, k[2]))
    t0, t1 = ks[0][0], max(k[1] for k in ks)
    busy = 0.0
    gaps = collections.defaultdict(lambda: [0, 0.0])
    cur_s, cur_e, last_name = ks[0][0], ks[0][1], ks[0][2]
    big = []
    for s, e, n in ks[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            g = s - cur_e
            if g >= a.min_gap_us:
                key = (short(last_name), short(n))
                gaps[key][0] += 1
                gaps[key][1] += g
                big.append((g, cur_e, key))
            cur_s, cur_e, last_name = s, e, n
        else:
            if e > cur_e:
                cur_e, last_name = e, n
    busy += cur_e - cur_s
    span = t1 - t0
    idle = span - busy
    ps = max(steps, 1)
    print(f"{a.trace.split('/')[-1]}: span {span / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms ({100 * busy / span:.1f}%), "
          f"idle {idle / 1e3:.1f} ms; steps (by {a.step_kernel!r}) {steps}: per step span {span / ps / 1e3:.2f} ms, "
          f"busy {busy / ps / 1e3:.2f} ms, idle {idle / ps / 1e3:.2f} ms")
    tot = sum(v[1] for v in gaps.values())
    print(f"gaps >= {a.min_gap_us:.0f} us: {sum(v[0] for v in gaps.values())} holding {tot / 1e3:.1f} ms "
          f"({100 * tot / max(idle, 1):.0f}% of idle); per step {tot / ps / 1e3:.2f} ms")
    print(f"{'n':>6} {'ms':>8} {'ms/step':>8} {'mean us':>8}  before -> after")
    for (b, aft), (n, g) in sorted(gaps.items(), key=lambda x: -x[1][1])[: a.top]:
        print(f"{n:6d} {g / 1e3:8.1f} {g / ps / 1e3:8.3f} {g / n:8.1f}  {b} -> {aft}")
    big.sort(reverse=True)
    print("largest gaps (ms):", ", ".join(f"{g / 1e3:.1f} [{k[0][:25]}->{k[1][:25]}]" for g, _, k in big[:8]))


if __name__ == "__main__":
    main()
