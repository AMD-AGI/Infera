#!/usr/bin/env python3
"""MoE + communication time per verify step, grouped by the step's largest DP-rank batch, for comparing the TP8 MoE
paths (production all-gather + prologue + gemm1/gemm2 + reduce-scatter against MegaMoeTP's one kernel) at equal rows.
kdeep.py cuts layers at the all-gather, which MegaMoeTP no longer launches; this only sums kernels by category.
Also prints the verify graph's heaviest kernels (to name new kernels). Uses kdeep.load.

Usage: moe_bybs.py [--layers 75] TRACE_RANK*.json.gz ...   (several profiles' files may be pooled)
"""
import argparse
import collections
import multiprocessing as mp
import re
import statistics as st
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import kdeep  # noqa: E402

# A kernel counts in the first category it matches (KF's fused gemm1 is named mfma_moe1_..._fp4q_sort_...).
CATS = [
    ("mega", r"mega", ),
    ("allgather", r"allgather|all_gather"),
    ("reduce_scatter", r"reduce_scatter"),
    ("allreduce", r"all_reduce|allreduce|cross_device_reduce"),
    ("moe_gemm1", r"gemm1_a4w4|mfma_moe1|flydsl_moe1|mxmoe_g1|moe1"),
    ("moe_gemm2", r"gemm2_a4w4|moe2"),
    ("moe_prologue", r"sort|P0_v2|P23|dynamic_per_group_scaled_quant|moe_align|append_shared_experts"),
    ("router_topk", r"topk|grouped_topk|biased_grouped"),
]


def category(name):
    for cat, rx in CATS:
        if re.search(rx, name, re.I):
            return cat
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=75)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("paths", nargs="+")
    a = ap.parse_args()
    groups = collections.defaultdict(list)
    for p in a.paths:
        groups[re.sub(r"-TP-\d+-DP-\d+.*", "", p)].append(p)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    heavy = collections.Counter()
    nsteps = 0
    with mp.Pool(min(16, len(a.paths))) as pool:
        for prof, files in groups.items():
            ranks = pool.map(kdeep.load, [(f, 0) for f in sorted(files)])
            vs = [[l for l in r["launches"] if l["kind"] == "verify"] for r in ranks]
            n = min(len(v) for v in vs)
            for i in range(n):
                mbs = max(v[i]["bs"] for v in vs)
                for v in vs:
                    ks = v[i]["k"]
                    row = acc[mbs]
                    row["kernel_sum"].append(sum(us for _, us in ks))
                    per = collections.Counter()
                    for k, us in ks:
                        per[category(k)] += us
                    for name, _ in CATS:
                        row[name].append(per[name])
                    for k, us in ks:
                        heavy[kdeep.short(k, 90)] += us
                    nsteps += 1
    print(f"{nsteps} rank-steps; per verify step, us (per MoE layer = / {a.layers} in brackets)")
    hdr = ["max_bs", "n", "kernel_sum"] + [n for n, _ in CATS]
    print("  ".join(f"{h:>14s}" for h in hdr))
    for mbs in sorted(acc):
        row = acc[mbs]
        cells = [f"{mbs:14d}", f"{len(row['kernel_sum']):14d}", f"{st.mean(row['kernel_sum']):14.0f}"]
        for name, _ in CATS:
            m = st.mean(row[name])
            cells.append(f"{m:8.0f}({m / a.layers:4.1f})")
        print("  ".join(cells))
    print(f"\nheaviest kernels in the verify graphs (us per rank-step):")
    for k, us in heavy.most_common(a.top):
        print(f"{us / nsteps:9.1f}  {k}")


if __name__ == "__main__":
    main()
