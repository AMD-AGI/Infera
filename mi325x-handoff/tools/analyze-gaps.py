#!/usr/bin/env python3
"""Where does the GPU idle time go, and what is the CPU doing during it?

analyze-trace.py answers "which kernels" -- useful while a single kernel dominates.
Once the profile flattens out, the interesting number is the *gaps*: at
concurrency 1 the decode leg is a chain of graph replays with CPU work between
them, and idle is the difference between what the GPU could do and what it is
asked to do.

Each gap is attributed to the CPU-side operator that spans it, which is what
distinguishes "the host cannot keep up" from "the GPU is waiting on a sync".
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict


def load(path: str) -> list[dict]:
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)["traceEvents"]


def main(path: str, top: int, min_us: float) -> int:
    events = load(path)

    # GPU kernels carry cat "kernel"; everything on a CPU thread is a host op.
    kernels, cpu_ops = [], []
    for e in events:
        if e.get("ph") != "X" or "dur" not in e:
            continue
        cat = e.get("cat", "")
        if cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels.append((e["ts"], e["ts"] + e["dur"], e.get("name", "")))
        elif cat in ("cpu_op", "user_annotation", "cuda_runtime", "hip_runtime"):
            cpu_ops.append((e["ts"], e["ts"] + e["dur"], e.get("name", ""), cat))

    if not kernels:
        print("no GPU kernels in trace")
        return 1

    kernels.sort()
    cpu_ops.sort()

    # Merge overlapping kernel intervals; concurrent streams must not be double
    # counted as busy, or the idle total comes out negative on a stream-heavy leg.
    merged = []
    cs, ce, _ = kernels[0]
    for s, e, _ in kernels[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    span = merged[-1][1] - merged[0][0]
    busy = sum(e - s for s, e in merged)
    gaps = [(merged[i + 1][0] - merged[i][1], merged[i][1], merged[i + 1][0])
            for i in range(len(merged) - 1)]
    gaps = [g for g in gaps if g[0] >= min_us]
    gaps.sort(reverse=True)

    print(f"span {span/1000:10.1f} ms")
    print(f"busy {busy/1000:10.1f} ms   ({100*busy/span:.1f}%)")
    print(f"idle {(span-busy)/1000:10.1f} ms   ({100*(span-busy)/span:.1f}%)")
    print(f"\ngaps >= {min_us:.0f} us: {len(gaps)}, "
          f"totalling {sum(g[0] for g in gaps)/1000:.1f} ms "
          f"({100*sum(g[0] for g in gaps)/span:.1f}% of span)")

    # Attribute each gap to the innermost CPU op covering its midpoint: the
    # outermost frame is always the module forward and says nothing.
    by_op: dict[str, list[float]] = defaultdict(list)
    for dur, gs, ge in gaps:
        mid = (gs + ge) / 2
        best = None
        for s, e, name, cat in cpu_ops:
            if s > mid:
                break
            if e >= mid and (best is None or (e - s) < best[0]):
                best = (e - s, name, cat)
        by_op[best[1] if best else "(no CPU op covering the gap)"].append(dur)

    print(f"\n--- gap time by covering CPU op (top {top}) ---")
    print(f"{'ms':>10} {'count':>7} {'us/gap':>9}  op")
    rows = sorted(by_op.items(), key=lambda kv: -sum(kv[1]))[:top]
    for name, ds in rows:
        print(f"{sum(ds)/1000:10.2f} {len(ds):7d} {sum(ds)/len(ds):9.1f}  {name[:88]}")

    print(f"\n--- largest individual gaps (top {min(top, len(gaps))}) ---")
    print(f"{'us':>10}  covering CPU op")
    for dur, gs, ge in gaps[:top]:
        mid = (gs + ge) / 2
        best = None
        for s, e, name, cat in cpu_ops:
            if s > mid:
                break
            if e >= mid and (best is None or (e - s) < best[0]):
                best = (e - s, name, cat)
        print(f"{dur:10.1f}  {best[1][:88] if best else '(none)'}")
    return 0


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("trace")
    a.add_argument("--top", type=int, default=15)
    a.add_argument("--min-us", type=float, default=20.0)
    a = a.parse_args()
    raise SystemExit(main(a.trace, a.top, a.min_us))
