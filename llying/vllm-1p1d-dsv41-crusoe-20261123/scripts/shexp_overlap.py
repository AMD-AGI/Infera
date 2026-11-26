#!/usr/bin/env python3
"""Do prefill's large FP8 GEMMs (the shared expert's among them) run beside RCCL? (EXPERIMENT 4.11, aux_prio)

For each trace: the large FP8 GEMMs (FlyDSL MXFP8 / AITER a8w8, at least MIN_US long) per hardware queue (the trace's
"stream"), and how much of their time overlaps an RCCL kernel on another queue. Without the overlap they all sit on the
main queue; with it, the shared expert's share moves to a side queue and runs under the MoE all-gather.
  shexp_overlap.py TRACE [TRACE ...]   (MIN_US env, default 100)
"""
import collections
import gzip
import json
import os
import re
import sys

MIN_US = float(os.environ.get("MIN_US", "100"))
GEMM = re.compile(r"flydsl_bpreshuffle_mxfp8|bmm_a8w8|mxscale|a8w8")


def main():
    for path in sys.argv[1:]:
        op = gzip.open if path.endswith(".gz") else open
        ks = [e for e in json.load(op(path, "rt"))["traceEvents"] if e.get("cat") == "kernel"]
        per_q = collections.Counter(k["tid"] for k in ks)
        main_q = per_q.most_common(1)[0][0]
        rccl = sorted((k["ts"], k["ts"] + k["dur"], k["tid"]) for k in ks if "ncclDevKernel" in k["name"])
        gemms = [k for k in ks if GEMM.search(k["name"]) and k["dur"] >= MIN_US]
        by_q = collections.defaultdict(lambda: [0, 0.0, 0.0])
        for g in gemms:
            s, e = g["ts"], g["ts"] + g["dur"]
            ov = sum(max(0.0, min(e, b) - max(s, a)) for a, b, q in rccl if q != g["tid"] and a < e and b > s)
            st = by_q[g["tid"]]
            st[0] += 1
            st[1] += g["dur"]
            st[2] += min(ov, g["dur"])
        print(f"== {path.split('/')[-1]}: main queue {main_q}; large FP8 GEMMs (>= {MIN_US:.0f} us):")
        for q, (n, d, o) in sorted(by_q.items()):
            print(f"  queue {q}{' (main)' if q == main_q else ''}: {n} GEMMs, {d / 1000:.1f} ms, "
                  f"{o / 1000:.1f} ms of it under RCCL on another queue ({100 * o / max(d, 1):.0f}%)")


if __name__ == "__main__":
    main()
