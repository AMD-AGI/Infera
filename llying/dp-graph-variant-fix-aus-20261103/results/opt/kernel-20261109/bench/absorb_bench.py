#!/usr/bin/env python3
"""GLM-5.2 MLA absorb BMMs on one DP-attention rank (64 heads), as sglang's forward_mla_rocm calls them:
  A = fused_fp8_bmm_rope_cat_and_cache_mla: q_nope (B, 64, 192) x W_kc -> (B, 64, 512) + RoPE(q_pe, k_pe) + cat
      + fp8 KV-cache write
  B = batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant: attn_out (B, 64, 512) x W_vc
      -> (B, 64, 256)
Each timed under CUDA-graph replay over NSETS weight sets (consecutive layers), per Triton config.

Usage (production image): absorb_bench.py --bs 48,72,88,96,128 [--out f.json]
"""
import argparse
import itertools
import json
import statistics as st

import torch

QH, P, R, DPE, V, POOL = 64, 192, 512, 64, 256, 1 << 20
FP8 = torch.float8_e4m3fn


def graph_time(fn, n_calls, reps=30):
    fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    t = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        t.append(a.elapsed_time(b) * 1e3 / n_calls)
    return st.median(t)


def configs():
    for bm, bn, nw, ns in itertools.product((16, 32, 64), (64, 128, 256), (4, 8), (1, 2)):
        yield {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "GROUP_SIZE_M": 1, "num_warps": nw, "num_stages": ns,
               "waves_per_eu": 1 if nw == 8 else 2, "matrix_instr_nonkdim": 16, "cache_modifier": ".cg"}


def snr(ref, x):
    ref, x = ref.float(), x.float()
    n = torch.mean((x - ref) ** 2).item()
    return 100.0 if n == 0 else 10 * torch.log10(torch.mean(ref ** 2) / n).item()


def main():
    from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
        batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant as bmm_vc,
    )
    from aiter.ops.triton.fusions.fused_bmm_rope_kv_cache import fused_fp8_bmm_rope_cat_and_cache_mla as bmm_kc

    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", default="48,72,88,96,128")
    ap.add_argument("--nsets", type=int, default=48)
    ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    w_k = [(torch.randn(QH, R, P, device="cuda") * 0.05).to(FP8) for _ in range(a.nsets)]   # w_kc^T, contiguous
    w_v = [(torch.randn(QH, V, R, device="cuda") * 0.05).to(FP8) for _ in range(a.nsets)]   # w_vc^T, contiguous
    w_scale = torch.tensor([1.0], device="cuda")
    kv_cache = torch.zeros(POOL, 1, R + DPE, device="cuda", dtype=FP8)
    max_pos = 1 << 18
    freqs = torch.randn(max_pos, DPE // 2, device="cuda")
    cos, sin = freqs.cos(), freqs.sin()
    k_scale = torch.tensor(1.0, device="cuda")
    rows = []
    for B in [int(x) for x in a.bs.split(",")]:
        q_nope = torch.randn(B, QH, P, device="cuda", dtype=torch.bfloat16)
        q_pe = torch.randn(B, QH, DPE, device="cuda", dtype=torch.bfloat16)
        k_nope = torch.randn(B, 1, R, device="cuda", dtype=torch.bfloat16)
        k_pe = torch.randn(B, 1, DPE, device="cuda", dtype=torch.bfloat16)
        slots = torch.randperm(POOL, device="cuda")[:B].to(torch.int64)
        pos = torch.randint(0, max_pos, (B,), device="cuda", dtype=torch.int64)
        attn_out = torch.randn(B, QH, R, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(B, QH, V, device="cuda", dtype=torch.bfloat16)

        def run_kc(cfg, ws=None):
            outs = []
            for w in (ws or w_k):
                outs.append(bmm_kc(q_nope.transpose(0, 1), w, w_scale, q_pe, k_nope, k_pe, kv_cache, slots, pos,
                                   cos, sin, group_size=128, k_scale=k_scale, is_neox=False, q_out_dtype=FP8,
                                   config=None if cfg is None else dict(cfg))[0])
            return outs

        def run_vc(cfg, ws=None):
            for w in (ws or w_v):
                bmm_vc(X=attn_out, WQ=w, w_scale=w_scale, group_size=128, YQ=y, transpose_bm=True,
                       transpose_bm_in=True, dtype=torch.bfloat16, config=None if cfg is None else dict(cfg))
            return y

        ref_kc = run_kc(None, w_k[:1])[0].float().clone()
        ref_vc = run_vc(None, w_v[:1]).float().clone()
        for name, run, ref, nbytes in (("kc", run_kc, ref_kc, QH * R * P), ("vc", run_vc, ref_vc, QH * V * R)):
            base = graph_time(lambda: run(None), a.nsets)
            best = (base, "default")
            for cfg in configs():
                try:
                    out = run(cfg, (w_k if name == "kc" else w_v)[:1])
                    out = (out[0] if isinstance(out, list) else out).float()
                    q = snr(ref, out)
                    if q < 40:
                        continue
                    us = graph_time(lambda: run(cfg), a.nsets)
                except Exception as e:  # noqa: BLE001
                    continue
                rows.append({"kernel": name, "B": B, "cfg": cfg, "us": us, "snr": q, "default_us": base})
                if us < best[0]:
                    best = (us, cfg)
            print(f"B {B:4d} {name}: default {base:6.2f} us ({nbytes / base / 1e6:.2f} TB/s)  best {best[0]:6.2f} us "
                  f"({(1 - best[0] / base) * 100:4.1f}% faster)  {best[1]}", flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
