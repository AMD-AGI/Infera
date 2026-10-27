#!/usr/bin/env python3
"""Which code issues the memcpys in a torch-profiler trace recorded with Python stacks: each gpu_memcpy of KIND
is linked to its HIP runtime call (correlation id), and the runtime call to the Python frames that enclose it on
the same thread; time and bytes are summed per innermost sglang/infera frame (with its caller) and per call chain.
KIND "kernel:SUBSTR" does the same for the kernels whose name contains SUBSTR.
Usage: prof_memcpy_callers.py TRACE.json.gz [KIND=HtoD] [TOP=15]
"""

import bisect
import gzip
import json
import sys
from collections import defaultdict


def main():
    path = sys.argv[1]
    kind = sys.argv[2] if len(sys.argv) > 2 else "HtoD"
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    t = json.load(gzip.open(path))
    ev = t["traceEvents"] if isinstance(t, dict) else t
    if kind.startswith("kernel:"):
        copies = [e for e in ev if e.get("cat") == "kernel" and kind[7:] in e.get("name", "")]
    else:
        copies = [e for e in ev if e.get("cat") == "gpu_memcpy" and e.get("args", {}).get("kind") == kind]
    by_corr = {e["args"]["correlation"]: e for e in copies}
    runtime = {e["args"]["correlation"]: e for e in ev
               if e.get("cat") == "cuda_runtime" and e.get("args", {}).get("correlation") in by_corr}
    frames = defaultdict(list)
    for e in ev:
        if e.get("cat") == "python_function":
            frames[e["tid"]].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"]))
    for tid in frames:
        frames[tid].sort()
    starts = {tid: [f[0] for f in fs] for tid, fs in frames.items()}

    def stack(tid, ts):
        fs, st = frames.get(tid, []), starts.get(tid, [])
        i = bisect.bisect_right(st, ts)
        enclosing = [f for f in fs[max(0, i - 4000):i] if f[1] >= ts]
        enclosing.sort(key=lambda f: f[0])
        return [f[2] for f in enclosing]

    def ours(name):
        return ("sglang/" in name or "infera/" in name) and "/torch/" not in name

    leaf = defaultdict(lambda: [0.0, 0, 0])
    chain = defaultdict(lambda: [0.0, 0, 0])
    total = [0.0, 0, 0]
    for corr, c in by_corr.items():
        r = runtime.get(corr)
        st = stack(r["tid"], r["ts"]) if r else []
        mine = [s for s in st if ours(s)]
        key = " <- ".join(reversed(mine[-2:])) if mine else (r["name"] if r else "no runtime event")
        ck = " > ".join(s.split("/")[-1] for s in mine[-6:]) if mine else key
        for d in (leaf[key], chain[ck], total):
            d[0] += c.get("dur", 0) / 1e3
            d[1] += c["args"].get("bytes", 0)
            d[2] += 1
    print(f"{path}: {kind} {total[2]} copies, {total[0]:.1f} ms, {total[1] / 2**20:.1f} MiB")
    print(f"-- by innermost sglang frame (<- caller):")
    for k, (ms, b, n) in sorted(leaf.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {ms:9.1f} ms {100 * ms / total[0]:5.1f}%  {b / 2**20:9.1f} MiB  n={n:5d}  {k}")
    print(f"-- by call chain (last 6 sglang frames):")
    for k, (ms, b, n) in sorted(chain.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"  {ms:9.1f} ms {100 * ms / total[0]:5.1f}%  {b / 2**20:9.1f} MiB  n={n:5d}  {k}")


if __name__ == "__main__":
    main()
