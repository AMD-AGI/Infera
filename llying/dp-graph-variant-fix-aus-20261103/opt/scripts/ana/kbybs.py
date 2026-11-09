#!/usr/bin/env python3
"""Per-call kernel time in the verify graph grouped by the step's largest DP-rank batch (all ranks replay the graph
captured for that batch), for before / after comparisons of single kernels at equal rows. Uses kdeep.load.

Usage: kbybs.py TRACE_RANK0.json.gz ...   (one profile; pass several profiles' files together to pool them)
"""
import collections
import multiprocessing as mp
import re
import statistics as st
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
import kdeep  # noqa: E402

PICK = [
    ("w_kc+rope", r"_fused_fp8_bmm_rope_cat_and_cache_mla"),
    ("w_vc", r"_batched_gemm_a8w8_a_per_token_group_prequant"),
    ("sparse_mla", r"_sparse_mla_split_k_kernel"),
    ("o_proj", r"Cijk_Alik_Bljk.*MT160x(96|128)x(64|128)"),
    ("moe_g1", r"gemm1_a4w4|mfma_moe1"),
    ("moe_g2", r"gemm2_a4w4"),
]


def main():
    paths = sys.argv[1:]
    groups = collections.defaultdict(list)
    for p in paths:
        groups[re.sub(r"-TP-\d+-DP-\d+.*", "", p)].append(p)
    acc = collections.defaultdict(lambda: collections.defaultdict(list))
    with mp.Pool(min(16, len(paths))) as pool:
        for prof, files in groups.items():
            ranks = pool.map(kdeep.load, [(f, 60e3) for f in sorted(files)])
            vs = [[l for l in r["launches"] if l["kind"] == "verify"] for r in ranks]
            n = min(len(v) for v in vs)
            for i in range(n):
                mbs = max(v[i]["bs"] for v in vs)
                for v in vs:
                    for name, rx in PICK:
                        d = [us for k, us in v[i]["k"] if re.search(rx, k)]
                        if d:
                            acc[mbs][name].append(st.mean(d))
    print("max_bs  steps*ranks  " + "  ".join(f"{n:>10s}" for n, _ in PICK) + "   (us per call)")
    for mbs in sorted(acc):
        row = acc[mbs]
        cnt = len(row[PICK[0][0]])
        print(f"{mbs:6d}  {cnt:11d}  " + "  ".join(f"{st.mean(row[n]) if row[n] else float('nan'):10.2f}" for n, _ in PICK))


if __name__ == "__main__":
    main()
