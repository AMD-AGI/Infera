#!/usr/bin/env python3
"""Decode-step bf16 GEMMs that read the same input, separate vs merged into one GEMM (the outputs are column slices of
the merged result): q_b_proj (16384 x 2048) + indexer wq_b (4096 x 2048), both on q_a; fused_qkv_a (2624 x 6144) +
indexer wk (128 x 6144) + weights_proj (32 x 6144), all on the hidden state (the indexer ones only in indexer layers).
aiter tgemm.mm with whatever tables are mounted / in AITER_CONFIG_GEMM_BF16; CUDA-graph us over 8 rotating weight sets.

Usage (1 GPU): python3 gemm_merge_bench.py [--ms 64,88,96] [--untuned OUT.csv]   (--untuned writes the shapes for the tuner)
"""
import argparse
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
from aklr_gemm_bench import graph_us  # noqa: E402

GROUPS = {
    "q_b+idx_wq_b": ([16384, 4096], 2048),
    "qkv_a+idx_wk+idx_wp": ([2624, 128, 32], 6144),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ms", default="64,88,96")
    ap.add_argument("--untuned", default=None)
    a = ap.parse_args()
    ms = [int(m) for m in a.ms.split(",")]
    if a.untuned:
        with open(a.untuned, "w") as f:
            f.write("M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n")
            for ns, k in GROUPS.values():
                for n in ns + [sum(ns)]:
                    for m in ms:
                        f.write(f"{m},{n},{k},False,torch.bfloat16,torch.bfloat16,False,False\n")
        print("wrote", a.untuned)
        return
    from aiter.tuned_gemm import tgemm

    g = torch.Generator(device="cuda").manual_seed(0)
    for name, (ns, k) in GROUPS.items():
        sets = [[torch.randn(n, k, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 for n in ns] for _ in range(8)]
        merged = [torch.cat(ws, 0).contiguous() for ws in sets]
        mb = sum(ns) * k * 2 / 1e6
        for m in ms:
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=g)
            ref = [tgemm.mm(x, w, None, torch.bfloat16) for w in sets[0]]
            out = tgemm.mm(x, merged[0], None, torch.bfloat16)
            offs = [0]
            for n in ns:
                offs.append(offs[-1] + n)
            err = max(((out[:, offs[i]:offs[i + 1]].float() - ref[i].float()).abs().max() / ref[i].float().abs().max()).item()
                      for i in range(len(ns)))
            parts = [graph_us(lambda w_i=i: [tgemm.mm(x, ws[w_i], None, torch.bfloat16) for ws in sets], len(sets))
                     for i in range(len(ns))]
            t_sep = graph_us(lambda: [[tgemm.mm(x, w, None, torch.bfloat16) for w in ws] for ws in sets], len(sets))
            t_mrg = graph_us(lambda: [tgemm.mm(x, w, None, torch.bfloat16) for w in merged], len(merged))
            print(f"{name:20s} M {m:3d}: separate {t_sep:6.1f} us (parts " + " + ".join(f"{p:.1f}" for p in parts)
                  + f"), merged {t_mrg:6.1f} us ({mb / t_mrg:4.2f} TB/s), saves {t_sep - t_mrg:5.1f} us, "
                  f"max rel diff vs separate {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
