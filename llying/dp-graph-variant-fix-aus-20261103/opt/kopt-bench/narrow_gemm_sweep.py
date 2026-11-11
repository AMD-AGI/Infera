#!/usr/bin/env python3
"""Narrow bf16 GEMMs of the decode step (per-rank rows 64-96): production aiter tgemm.mm (with the production tuned
tables mounted under aiter/configs/model_configs/) against aiter's Triton GEMMs over a config grid:
  atomic:  gemm_a16w16_atomic, split-K partials added atomically into a zeroed bf16 output (zeroing timed too);
  splitk:  gemm_a16w16, split-K with its own reduce kernel.
CUDA-graph us over 8 rotating weights; error = max |y - y_fp32| / max |y_fp32| against an fp32 reference, next to the
production kernel's own error. Prints the best few configs per variant and M.

Usage (1 GPU, production image + tables): python3 narrow_gemm_sweep.py --shape fused_qkv_a [--ms 64,88,96]
"""
import argparse
import itertools
import json
import sys

import torch

sys.path.insert(0, "/perf_apps/liyingli/bench_agentx/kopt-20261109/bench")
from aklr_gemm_bench import graph_us  # noqa: E402

SHAPES = {"fused_qkv_a": (2624, 6144), "q_b_proj": (16384, 2048), "idx_wq_b": (4096, 2048), "o_proj": (6144, 16384)}


def err(y, ref):
    return ((y.float() - ref).abs().max() / ref.abs().max()).item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=sorted(SHAPES))
    ap.add_argument("--ms", default="64,88,96")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    N, K = SHAPES[a.shape]
    ms = [int(m) for m in a.ms.split(",")]
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
    from aiter.tuned_gemm import tgemm

    g = torch.Generator(device="cuda").manual_seed(0)
    ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g) * 0.02 for _ in range(8)]
    xs = {M: torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g) for M in ms}
    refs = {M: xs[M].float() @ ws[0].float().t() for M in ms}
    gb = N * K * 2 / 1e3
    base = {}
    for M in ms:
        x = xs[M]
        t = graph_us(lambda: [tgemm.mm(x, w, None, torch.bfloat16) for w in ws], len(ws))
        base[M] = (t, err(tgemm.mm(x, ws[0], None, torch.bfloat16), refs[M]))
        print(f"{a.shape} M {M}: production {t:6.1f} us ({gb / t / 1e3:4.2f} TB/s), err {base[M][1]:.2e}", flush=True)

    grid = []
    for bm, bn, bk, ks, nw, wpe in itertools.product((32, 128), (32, 64, 128), (128, 256), (2, 4, 8, 16), (4, 8), (1,)):
        if K // ks < bk or K % (ks * bk):
            continue
        grid.append({"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1, "num_warps": nw,
                     "num_stages": 2, "waves_per_eu": wpe, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg",
                     "NUM_KSPLIT": ks})
    results = []
    ys = {M: torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for M in ms}
    for variant in ("atomic", "splitk"):
        for cfg in grid:
            row = {"variant": variant, "cfg": cfg}
            try:
                for M in ms:
                    x, y = xs[M], ys[M]
                    if variant == "atomic":
                        def call(w, x=x, y=y):
                            y.zero_()
                            gemm_a16w16_atomic(x, w, torch.bfloat16, y, config=dict(cfg))
                    else:
                        def call(w, x=x, y=y):
                            gemm_a16w16(x, w, None, torch.bfloat16, y, config=dict(cfg))
                    call(ws[0])
                    e = err(y, refs[M])
                    t = graph_us(lambda: [call(w) for w in ws], len(ws))
                    row[M] = (round(t, 2), e)
            except Exception as ex:  # noqa: BLE001
                row["error"] = f"{type(ex).__name__}: {str(ex)[:120]}"
            results.append(row)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"shape": a.shape, "base": base, "results": results}, f)
    for variant in ("atomic", "splitk"):
        for M in ms:
            ok = [r for r in results if r["variant"] == variant and M in r and r[M][1] < max(4 * base[M][1], 1e-2)]
            ok.sort(key=lambda r: r[M][0])
            print(f"{a.shape} {variant} M {M}: best of {len(ok)} accurate configs (production {base[M][0]:.1f} us):",
                  flush=True)
            for r in ok[:3]:
                c = r["cfg"]
                print(f"    {r[M][0]:6.1f} us ({gb / r[M][0] / 1e3:4.2f} TB/s, err {r[M][1]:.1e})  "
                      f"BM{c['BLOCK_SIZE_M']} BN{c['BLOCK_SIZE_N']} BK{c['BLOCK_SIZE_K']} KS{c['NUM_KSPLIT']} "
                      f"w{c['num_warps']} wpe{c['waves_per_eu']}", flush=True)
    nerr = sum("error" in r for r in results)
    print(f"{a.shape}: {len(results)} runs, {nerr} failed", flush=True)


if __name__ == "__main__":
    main()
