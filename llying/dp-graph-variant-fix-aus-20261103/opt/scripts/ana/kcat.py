#!/usr/bin/env python3
"""Refined verify-graph categories from kdeep.py --json outputs (ms/step), with achieved bandwidth for the
weight-streaming kernels. GLM-5.2-MXFP4, attention DP (full attention weights per rank), MoE TP over all ranks.

Usage: kcat.py --tp 8 A.json [B.json ...]   (several trace sets of one config are averaged)
"""
import argparse
import collections
import json
import re

CATS = [
    ("moe_gemm1", r"gemm1_a4w4|mfma_moe1|flydsl_moe1|mxmoe_g1"),
    ("moe_gemm2", r"gemm2_a4w4|moe2_"),
    ("moe_reduce", r"moe_reduction"),
    ("moe_router_gemm", r"hgemm_bf16_t64x64x64x6_ksd_w4x2x1|hgemm_bf16_t32x64x128x4_ksd_w2x2x1"),
    ("moe_prologue", r"topk_reg|append_shared|opus_moe_sorting|dynamic_per_group_scaled_quant|mxfp4_moe_sort|moe_sort_fill"),
    ("proj_qkv_a", r"hgemm_bf16_t64x64x64x6_ksd_w2x4x1"),
    ("proj_q_b", r"hgemm_bf16_t128x64x64x4_ks1_w4x1x2"),
    ("proj_o", r"Cijk_Alik_Bljk.*MT160x(96|128)x(64|128)|PostGSU8"),
    ("mla_absorb_bmm", r"_fused_fp8_bmm_rope_cat_and_cache_mla|_batched_gemm_a8w8"),
    ("sparse_mla", r"_sparse_mla_"),
    ("indexer_logits", r"paged_mqa_logits"),
    ("indexer_topk", r"_topk_est|_topk_collect|_topk_select"),
    ("indexer_prep", r"hgemm_bf16_t64x32x128x4|hgemm_bf16_t16x32x128x[67]|Layernorm2dFwd|kn_entry_2c|hadamard|act_quant_kernel|indexer_k_quant|triton_poi_fused_mul"),
    ("norm", r"rmsnorm"),
    ("comm", r"allgather|reduce_scatter|ncclDevKernel|cross_device_reduce"),
    ("dense_mlp_l0_2", r"MT256x144x64|MT192x96x128|MT128x128x256|act_and_mul|PostGSU4|hgemm_bf16_t128x96x64x4|hgemm_bf16_t96x96x64x5"),
    ("lm_head_tail", r"MT256x240x64|MT192x320x64|MT320x192x64|MT160x224x64"),
]

MB = 1e6
# Weight bytes per call (per rank). MXFP4 = 0.5 B/elem + one E8M0 byte per 32 elements.
def mxfp4(n):
    return n * 0.5 * (1 + 1 / 16)


def weights(tp):
    inter = 2048 // tp
    return {
        "moe_gemm1": (75, mxfp4(257 * 2 * inter * 6144)),
        "moe_gemm2": (75, mxfp4(257 * 6144 * inter)),
        "moe_router_gemm": (75, 256 * 6144 * 2),
        "proj_qkv_a": (78, 2624 * 6144 * 2),
        "proj_q_b": (78, 16384 * 2048 * 2),
        "proj_o": (78, 6144 * 16384 * 2),
        "mla_absorb_bmm": (78, 64 * (192 * 512 + 512 * 256)),
    }


def cat_of(name):
    for c, rx in CATS:
        if re.search(rx, name):
            return c
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, required=True)
    ap.add_argument("jsons", nargs="+")
    a = ap.parse_args()
    acc = collections.Counter()
    other = collections.Counter()
    for p in a.jsons:
        d = json.load(open(p))
        for k, ms in d["kernel_ms_per_step"].items():
            c = cat_of(k)
            acc[c] += ms / len(a.jsons)
            if c == "other":
                other[k[:90]] += ms / len(a.jsons)
    w = weights(a.tp)
    total = sum(acc.values())
    print(f"TP{a.tp}: verify graph {total:.2f} ms/step (mean of {len(a.jsons)} trace sets)")
    for c, _ in CATS + [("other", "")]:
        ms = acc[c]
        extra = ""
        if c in w:
            n, b = w[c]
            extra = f"  {n * b / 1e9:6.2f} GB -> {n * b / (ms * 1e-3) / 1e12:5.2f} TB/s  ({b / MB:6.1f} MB/call)"
        print(f"  {c:16s} {ms:6.2f}{extra}")
    for k, ms in other.most_common(6):
        print(f"     other: {ms:.3f}  {k}")


if __name__ == "__main__":
    main()
