#!/usr/bin/env python3
"""lm_head GEMM of the TP8 decode graphs (vocab-parallel: N = 154880 / 8 = 19360, K = 6144, bf16; M = rows gathered
over the 8 DP ranks): production torch.matmul (hipBLASLt default heuristic) against aiter tgemm.mm with the tuned
table in AITER_CONFIG_GEMM_BF16. CUDA-graph us over 8 rotating weights, and the max relative error vs torch.

Usage (1 GPU): AITER_CONFIG_GEMM_BF16=TUNED.csv python3 lmhead_bench.py --ms 96,192,384,576,768
"""
import argparse
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
from aklr_gemm_bench import graph_us  # noqa: E402

N, K = 19360, 6144


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="96,192,384,576,768")
    a = ap.parse_args()
    from aiter.tuned_gemm import tgemm

    g = torch.Generator(device="cuda").manual_seed(0)
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 for _ in range(8)]
    for M in [int(m) for m in a.ms.split(",")]:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
        ref = torch.matmul(x, ws[0].t()).float()
        got = tgemm.mm(x, ws[0], None, torch.bfloat16).float()
        err = ((got - ref).abs().max() / ref.abs().max()).item()
        tt = graph_us(lambda: [torch.matmul(x, w.t()) for w in ws], len(ws))
        ta = graph_us(lambda: [tgemm.mm(x, w, None, torch.bfloat16) for w in ws], len(ws))
        tb = N * K * 2 / 1e6
        print(f"M {M:4d}: torch.matmul {tt:6.1f} us ({tb / tt:4.2f} TB/s, {2 * M * N * K / tt / 1e6:5.0f} TFLOP/s)  "
              f"aiter tuned {ta:6.1f} us ({tb / ta:4.2f} TB/s)  speedup {tt / ta:4.2f}x  max rel err {err:.2e}", flush=True)


if __name__ == "__main__":
    main()
