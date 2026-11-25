#!/usr/bin/env python3
"""Stream overlap and per-call kernel times in one torch-profiler trace (Chrome JSON, .json or .json.gz).

For the shared-expert stream overlap (EXPERIMENT 4.11, R15): GPU time per stream; how much of the non-main streams'
kernel time runs while an RCCL kernel (ncclDevKernel) runs on the main stream; and the mean / p50 duration per call
of the communication, routed-MoE and FP8 dense GEMM kernels, so two traces can be compared call for call.
  trace_overlap.py TRACE [TRACE ...]
"""
import gzip
import json
import re
import statistics
import sys
from collections import defaultdict

GROUPS = [
    ("rccl", r"ncclDevKernel"),
    ("moe_gemm1", r"^gemm1_a4w4"),
    ("moe_gemm2", r"^gemm2_a4w4"),
    ("fp8_gemm_k2304 (shared expert down)", r"k2304"),
    ("fp8_gemm_other", r"flydsl_bpreshuffle_mxfp8|flydsl_mxpsh|bmm_a8w8"),
    ("sparse_mla", r"sparse_mla"),
]


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [e for e in json.load(f)["traceEvents"] if e.get("cat") == "kernel"]


def merged(intervals):
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a, b):
    """Total length of the intersection of two merged interval lists."""
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if lo < hi:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def report(path):
    ks = load(path)
    by_stream = defaultdict(list)
    for k in ks:
        by_stream[k.get("tid")].append(k)
    main = max(by_stream, key=lambda t: sum(k["dur"] for k in by_stream[t]))
    span = (max(k["ts"] + k["dur"] for k in ks) - min(k["ts"] for k in ks)) / 1000
    print(f"== {path.split('/')[-1]}: {len(ks)} kernels over {span:.0f} ms; main stream {main}")
    for t, v in sorted(by_stream.items(), key=lambda kv: -sum(k["dur"] for k in kv[1])):
        print(f"  stream {t}: {len(v)} kernels, {sum(k['dur'] for k in v) / 1000:.1f} ms")
    rccl = merged([(k["ts"], k["ts"] + k["dur"]) for k in by_stream[main] if "ncclDevKernel" in k["name"]])
    side = merged([(k["ts"], k["ts"] + k["dur"]) for t, v in by_stream.items() if t != main for k in v])
    side_ms = sum(e - s for s, e in side) / 1000
    print(f"  non-main streams busy {side_ms:.1f} ms, of which {overlap(side, rccl) / 1000:.1f} ms while RCCL runs")
    for name, rx in GROUPS:
        d = [k["dur"] / 1000 for k in ks if re.search(rx, k["name"])]
        if d:
            print(f"  {name:40s} {len(d):6d} calls, total {sum(d):8.1f} ms, mean {statistics.mean(d):.3f} "
                  f"p50 {statistics.median(d):.3f} ms")


for p in sys.argv[1:]:
    report(p)
