#!/usr/bin/env python3
"""K4: the TP8 decode dense projections (fused_qkv_a 2624x6144, q_b_proj 16384x2048, o_proj 6144x16384, bf16) at our
per-rank rows, production against aiter's Triton split-K kernels with cold weights.

  prod     aiter tgemm.mm with the production tuned CSVs mounted (what sglang's unquantized linear calls on ROCm)
  splitk   Triton gemm_a16w16: split-K partials in fp32 + a reduce kernel (two launches)
  atomic   Triton gemm_a16w16_atomic: split-K partials atomically added into the bf16 output, zeroed first (memset +
           one launch); atomic_acc = the same into an existing buffer without the memset (what fusing the residual
           add, y += x @ w.T, would cost)

Weights rotate over enough copies (>= 1 GB) that every call reads them from HBM, as in the decode where every layer has
its own (o_proj's 201 MB would otherwise sit in the 256 MB MALL). CUDA-graph replay of NSETS calls, us per call.
The config sweep is split over processes (--part i/n, one GPU each); --merge prints the best per shape and M.

Usage (production image, one process per GPU): k4_splitk_sweep.py --part 3/8 --out DIR
       k4_splitk_sweep.py --merge DIR
"""
import argparse
import glob
import itertools
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aklr_gemm_bench import graph_us  # noqa: E402

SHAPES = {"fused_qkv_a": (2624, 6144), "q_b_proj": (16384, 2048), "o_proj": (6144, 16384)}
MS = (32, 48, 64, 80, 96, 128)


def configs(K):
    for bm, bn, bk, ks, nw in itertools.product((16, 32, 64), (32, 64, 128), (128, 256), (1, 2, 4, 8, 16), (4, 8)):
        if K // ks < bk:
            continue
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk, "GROUP_SIZE_M": 1, "NUM_KSPLIT": ks,
               "num_warps": nw, "num_stages": 2, "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}


def sweep(part, nparts, out_dir):
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
    from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
    from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
    from aiter.tuned_gemm import tgemm

    out = open(os.path.join(out_dir, f"part{part}.jsonl"), "w")
    torch.manual_seed(0)
    for name, (N, K) in SHAPES.items():
        nbytes = N * K * 2
        nsets = max(4, -(-(1 << 30) // nbytes))
        ws = [torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.02 for _ in range(nsets)]
        xs = {M: torch.randn(M, K, device="cuda", dtype=torch.bfloat16) for M in MS}
        refs = {M: xs[M].float() @ ws[0].float().t() for M in MS}
        ys = {M: [torch.empty(M, N, device="cuda", dtype=torch.bfloat16) for _ in ws] for M in MS}

        def rec(kind, M, cfg, fn, check):
            try:
                us = graph_us(fn, nsets)
                o = check()
                err = ((o.float() - refs[M]).abs().max() / refs[M].abs().max()).item()
            except Exception as e:  # noqa: BLE001
                us, err = None, f"{type(e).__name__}: {str(e)[:100]}"
            out.write(json.dumps({"shape": name, "M": M, "kind": kind, "cfg": cfg, "us": us, "err": err,
                                  "tbps": nbytes / us / 1e6 if us else None}) + "\n")
            out.flush()

        if part == 0:
            for M in MS:
                x = xs[M]
                rec("prod", M, None, lambda: [tgemm.mm(x, w, None, torch.bfloat16) for w in ws],
                    lambda: tgemm.mm(x, ws[0], None, torch.bfloat16))
        for i, base in enumerate(configs(K)):
            if i % nparts != part:
                continue
            for M in MS:
                x, yl = xs[M], ys[M]
                if base["BLOCK_SIZE_M"] > max(16, 1 << (M - 1).bit_length()):
                    continue
                c = compute_splitk_params(dict(base), K)
                rec("splitk", M, base,
                    lambda: [gemm_a16w16(x, w, None, torch.bfloat16, y, config=dict(c)) for w, y in zip(ws, yl)],
                    lambda: gemm_a16w16(x, ws[0], None, torch.bfloat16, yl[0], config=dict(c)))
                a = {k: v for k, v in base.items()}

                def atomic(zero):
                    def fn():
                        for w, y in zip(ws, yl):
                            if zero:
                                y.zero_()
                            gemm_a16w16_atomic(x, w, torch.bfloat16, y, config=dict(a), accumulate=not zero)
                    return fn

                def atomic_check():
                    yl[0].zero_()
                    return gemm_a16w16_atomic(x, ws[0], torch.bfloat16, yl[0], config=dict(a), accumulate=True)

                rec("atomic", M, base, atomic(True), atomic_check)
                rec("atomic_acc", M, base, atomic(False), atomic_check)
        del ws, xs, ys, refs
        torch.cuda.empty_cache()
    out.close()


def merge(out_dir):
    rows = [json.loads(line) for f in sorted(glob.glob(os.path.join(out_dir, "part*.jsonl"))) for line in open(f)]
    ok = [r for r in rows if r["us"] and isinstance(r["err"], float) and r["err"] < 2e-2]
    bad = sum(1 for r in rows if r["us"] and not (isinstance(r["err"], float) and r["err"] < 2e-2))
    fail = sum(1 for r in rows if not r["us"])
    print(f"{len(rows)} measurements, {len(ok)} valid, {bad} wrong results, {fail} failed to compile / run")
    for name in SHAPES:
        print(f"\n== {name} {SHAPES[name]}")
        for M in MS:
            prod = [r for r in ok if r["shape"] == name and r["M"] == M and r["kind"] == "prod"]
            p = prod[0]["us"] if prod else float("nan")
            line = f"M {M:4d}: prod {p:6.1f} us ({prod[0]['tbps'] if prod else 0:4.2f} TB/s)"
            for kind in ("splitk", "atomic", "atomic_acc"):
                c = [r for r in ok if r["shape"] == name and r["M"] == M and r["kind"] == kind]
                if c:
                    b = min(c, key=lambda r: r["us"])
                    g = b["cfg"]
                    line += (f" | {kind} {b['us']:6.1f} ({p / b['us']:4.2f}x, err {b['err']:.1e}, "
                             f"{g['BLOCK_SIZE_M']}x{g['BLOCK_SIZE_N']}x{g['BLOCK_SIZE_K']} ks{g['NUM_KSPLIT']} w{g['num_warps']})")
            print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--out")
    ap.add_argument("--merge")
    a = ap.parse_args()
    if a.merge:
        merge(a.merge)
    else:
        i, n = (int(v) for v in a.part.split("/"))
        os.makedirs(a.out, exist_ok=True)
        sweep(i, n, a.out)
