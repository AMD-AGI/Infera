###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Kernel tasks: a replay's kernels, written for an agent that will rewrite them.

:mod:`kernel_shapes` reports every kernel a replay launches, one row per shape.
This regroups those rows into one task per kernel family (pool, op, dtype) and
adds what an agent writing or tuning that kernel needs to start without a
profiler trace:

* what the kernel computes, its tensors with their shapes, dtypes and
  layouts, and a PyTorch reference to check a rewrite against;
* the concrete shape cases the trace runs, heaviest first, with each case's
  share of the calls and of the time;
* a time estimate per case and where it came from, the speed-of-light time
  (the kernel's bytes and FLOPs at the GPU's peak bandwidth and compute), and
  the headroom between them;
* the family's share of the replay's GPU time, and what the deployment would
  gain end to end if the family ran at its speed of light (Amdahl);
* the deployment it was derived for (model geometry, sharding, dtypes,
  engine), the library kernels the engines usually dispatch it to, and the
  keys their tuners index by.

The engine kernels are hints. Which one a deployment dispatches depends on the
engine version and flags; check the serving stack before replacing one.

The JSON (``inferasim.kernel_tasks``) is the machine-readable form; the
Markdown rendering is the same content for a prompt.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
import time
from collections import defaultdict

from . import gemm_shapes
from . import kernel_shapes as ks

SCHEMA = "inferasim.kernel_tasks"
VERSION = 1

# Cases listed per task: enough to cover this share of its time, at most this many.
CASE_COVERAGE = 0.95
MAX_CASES = 24

_TOLERANCE = {
    "fp32": {"atol": 1e-5, "rtol": 1e-5},
    "bf16": {"atol": 1e-2, "rtol": 1.6e-2},
    "fp16": {"atol": 1e-3, "rtol": 1e-3},
    "fp8": {"atol": 6e-2, "rtol": 1.25e-1, "note": "or SNR >= 30 dB against the bf16 reference"},
    "mxfp4": {"atol": 2.5e-1, "rtol": 2.5e-1, "note": "or SNR >= 20 dB against the bf16 reference"},
    "int32": {"exact": True},
}

# Peak-compute multiplier over bf16 by the GEMM's operand precision.
_ELEM_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1, "mxfp4": 0.5, "fp4": 0.5}
_COMPUTE_X = {"fp8": 2.0, "int8": 2.0, "mxfp4": 4.0, "fp4": 4.0, "mxfp6": 4.0}


# ── families ────────────────────────────────────────────────────────────────


def _family(kind: str, op: str, dims: dict) -> str:
    if kind == "gemm":
        if dims.get("groups", 1) > 1 and dims.get("topk", 0) > 0:
            return "moe_grouped_gemm"
        if op in ("attn_q_absorb", "attn_o_absorb"):
            return "batched_gemm"
        if dims.get("groups", 1) > 1:
            return "grouped_gemm"
        return "gemm"
    if kind == "attention":
        if dims.get("compress"):
            return "v4_attention"
        return op  # attn_prefill / attn_decode
    if kind == "indexer":
        return (
            "indexer_topk"
            if op == "indexer_topk"
            else ("index_cache_write" if op == "index_cache_write" else "indexer_score")
        )
    if kind == "linear_attention":
        if op == "causal_conv1d":
            return "causal_conv1d"
        return "linear_attention_decode" if op.endswith("_decode") else "linear_attention_prefill"
    if kind == "quant":
        return "act_quant"
    if kind == "collective":
        return "collective"
    if op in ("kv_cache_write", "pool_cache_write"):
        return "kv_cache_write"
    if op.endswith("silu_and_mul") or op.endswith("gelu"):
        return "act_and_mul"
    if kind == "norm":
        return (
            "gated_rmsnorm"
            if op == "gated_rmsnorm"
            else ("qk_rmsnorm" if op == "qk_rmsnorm" else "rmsnorm")
        )
    return op


# Per family: what it computes, its tensors, its reference, engine kernels, tuner keys.
# Shapes are written in the case's dimension names (the columns of the kernel report).
_FAMILIES: dict[str, dict] = {
    "gemm": {
        "computes": "C = A @ B^T: a linear layer over the step's tokens on one rank "
        "(nn.Linear layout, weight stored [N, K]).",
        "inputs": [
            {"name": "A", "shape": ["m", "k"], "dtype": "$act", "layout": "row-major"},
            {"name": "B", "shape": ["n", "k"], "dtype": "$dtype", "layout": "row-major (NT)"},
        ],
        "outputs": [{"name": "C", "shape": ["m", "n"], "dtype": "bf16"}],
        "reference": "def ref(A, B):\n    return (A.float() @ B.float().t()).to(torch.bfloat16)",
        "engines": {
            "vllm": [
                "hipBLASLt via torch (bf16)",
                "aiter gemm_a8w8 / gemm_a8w8_blockscale (fp8)",
                "aiter / triton w4a16 or mxfp4 GEMM (fp4)",
            ],
            "sglang": [
                "hipBLASLt via torch (bf16)",
                "aiter gemm_a16w16 (bf16)",
                "aiter gemm_a8w8_blockscale (fp8)",
            ],
        },
        "tuning_keys": ["m", "n", "k", "dtype"],
    },
    "moe_grouped_gemm": {
        "computes": "Fused MoE expert GEMM over this rank's local experts: each routed "
        "(token, expert) row is multiplied by its expert's weight. m is the tokens "
        "entering the fused-MoE op (the key vLLM and AITER tune by), rows_per_expert "
        "the mean rows one local expert sees.",
        "inputs": [
            {"name": "hidden", "shape": ["m", "k"], "dtype": "$act", "layout": "row-major"},
            {
                "name": "w",
                "shape": ["groups", "n", "k"],
                "dtype": "$dtype",
                "layout": "per expert row-major (NT)",
            },
            {"name": "topk_ids", "shape": ["m", "topk"], "dtype": "int32"},
            {"name": "topk_weights", "shape": ["m", "topk"], "dtype": "fp32"},
        ],
        "outputs": [{"name": "out", "shape": ["m", "topk", "n"], "dtype": "bf16"}],
        "reference": (
            "def ref(hidden, w, topk_ids):\n"
            "    # topk_ids index local experts; rows routed elsewhere are skipped\n"
            "    out = torch.zeros(*topk_ids.shape, w.shape[1], dtype=torch.float32)\n"
            "    for e in range(w.shape[0]):\n"
            "        t, j = (topk_ids == e).nonzero(as_tuple=True)\n"
            "        out[t, j] = hidden[t].float() @ w[e].float().t()\n"
            "    return out.to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": ["aiter fused_moe (CK / asm 2-stage)", "triton fused_moe (fused_moe_kernel)"],
            "sglang": ["aiter fused_moe", "triton fused_moe"],
        },
        "tuning_keys": ["m", "n", "k", "groups", "topk", "dtype"],
    },
    "grouped_gemm": {
        "computes": "Grouped linear: the groups' inputs are disjoint column slices of A, "
        "each multiplied by its own weight (DeepSeek-V4's grouped output LoRA).",
        "inputs": [
            {"name": "A", "shape": ["rows_per_expert", "groups", "k"], "dtype": "$act"},
            {"name": "B", "shape": ["groups", "n", "k"], "dtype": "$dtype", "layout": "NT"},
        ],
        "outputs": [{"name": "C", "shape": ["rows_per_expert", "groups", "n"], "dtype": "bf16"}],
        "reference": (
            "def ref(A, B):\n"
            "    return torch.einsum('tgk,gnk->tgn', A.float(), B.float()).to(torch.bfloat16)"
        ),
        "engines": {"vllm": ["torch.bmm / einsum (hipBLASLt batched)"], "sglang": ["torch.bmm"]},
        "tuning_keys": ["rows_per_expert", "n", "k", "groups", "dtype"],
    },
    "batched_gemm": {
        "computes": "MLA's absorbed decode projections, one batch per local head: "
        "attn_q_absorb maps the query's no-RoPE part into the latent (q_nope @ W_UK), "
        "attn_o_absorb maps the latent attention output back to the value dim (o @ W_UV).",
        "inputs": [
            {"name": "A", "shape": ["groups", "rows_per_expert", "k"], "dtype": "bf16"},
            {"name": "B", "shape": ["groups", "k", "n"], "dtype": "bf16", "layout": "NN"},
        ],
        "outputs": [{"name": "C", "shape": ["groups", "rows_per_expert", "n"], "dtype": "bf16"}],
        "reference": "def ref(A, B):\n    return torch.bmm(A.float(), B.float()).to(torch.bfloat16)",
        "engines": {
            "vllm": ["torch.bmm (hipBLASLt)", "aiter batched_gemm_a8w8 (fp8 MLA)"],
            "sglang": ["torch.bmm", "aiter batched GEMM"],
        },
        "tuning_keys": ["groups", "rows_per_expert", "n", "k"],
    },
    "attn_decode": {
        "computes": "Paged decode attention: each of batch sequences attends its q new "
        "tokens over its KV cache. Spans are bucketed; max_kv and mean_kv include this "
        "step's tokens. MLA decode is the absorbed form: kv_heads = 1, qk_dim = "
        "kv_lora_rank + rope over the shared latent cache, v_dim = kv_lora_rank. "
        "window > 0 limits each query to the last window keys; topk > 0 attends only the "
        "indexer's top-k keys.",
        "inputs": [
            {"name": "q", "shape": ["batch", "q", "heads", "qk_dim"], "dtype": "bf16"},
            {
                "name": "kv_cache",
                "shape": ["num_blocks", "$block_size", "kv_heads", "qk_dim"],
                "dtype": "$dtype",
                "layout": "paged; V is the first v_dim channels for MLA, "
                "a separate cache otherwise",
            },
            {
                "name": "block_table",
                "shape": ["batch", "ceil(max_kv / $block_size)"],
                "dtype": "int32",
            },
            {"name": "seq_lens", "shape": ["batch"], "dtype": "int32"},
        ],
        "outputs": [{"name": "out", "shape": ["batch", "q", "heads", "v_dim"], "dtype": "bf16"}],
        "reference": (
            "def ref(q, k, v, seq_lens, scale, window=0):\n"
            "    # k, v: [batch, max_kv, kv_heads, dim] gathered from the paged cache\n"
            "    out = []\n"
            "    for b, L in enumerate(seq_lens.tolist()):\n"
            "        kb, vb = k[b, :L].float(), v[b, :L].float()\n"
            "        g = q.shape[2] // kb.shape[1]\n"
            "        kb, vb = kb.repeat_interleave(g, 1), vb.repeat_interleave(g, 1)\n"
            "        s = torch.einsum('qhd,khd->hqk', q[b].float(), kb) * scale\n"
            "        pos = torch.arange(L - q.shape[1], L)[:, None]\n"
            "        key = torch.arange(L)[None, :]\n"
            "        mask = key > pos\n"
            "        if window:\n"
            "            mask |= key <= pos - window\n"
            "        s = s.masked_fill(mask, float('-inf'))\n"
            "        out.append(torch.einsum('hqk,khd->qhd', s.softmax(-1), vb))\n"
            "    return torch.stack(out).to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": [
                "aiter paged_attention / pa_fwd (ROCM_AITER_FA)",
                "aiter mla_decode_fwd (MLA)",
                "triton unified_attention (TRITON_ATTN)",
            ],
            "sglang": [
                "aiter mla_decode_fwd (MLA)",
                "aiter pa_fwd (aiter backend)",
                "aiter triton unified_attention",
                "triton decode_attention",
            ],
        },
        "tuning_keys": [
            "batch",
            "q",
            "heads",
            "kv_heads",
            "qk_dim",
            "max_kv",
            "dtype",
            "kv_block_size",
        ],
    },
    "attn_prefill": {
        "computes": "Variable-length causal prefill (extend) attention: batch sequences "
        "with q new tokens in total (max_q the longest) attend over their cached prefix "
        "plus themselves (max_kv, mean_kv). MLA prefill runs per head with qk_dim = "
        "qk_nope + rope and v_dim = v_head_dim.",
        "inputs": [
            {
                "name": "q",
                "shape": ["q", "heads", "qk_dim"],
                "dtype": "bf16",
                "layout": "packed varlen",
            },
            {"name": "k", "shape": ["sum(kv)", "kv_heads", "qk_dim"], "dtype": "bf16"},
            {"name": "v", "shape": ["sum(kv)", "kv_heads", "v_dim"], "dtype": "bf16"},
            {"name": "cu_seqlens_q", "shape": ["batch + 1"], "dtype": "int32"},
            {"name": "cu_seqlens_k", "shape": ["batch + 1"], "dtype": "int32"},
        ],
        "outputs": [{"name": "out", "shape": ["q", "heads", "v_dim"], "dtype": "bf16"}],
        "reference": (
            "def ref(q, k, v, cu_q, cu_k, scale, window=0):\n"
            "    out = torch.empty(q.shape[0], q.shape[1], v.shape[2])\n"
            "    for b in range(len(cu_q) - 1):\n"
            "        qs, ks_ = q[cu_q[b]:cu_q[b + 1]].float(), k[cu_k[b]:cu_k[b + 1]].float()\n"
            "        vs = v[cu_k[b]:cu_k[b + 1]].float()\n"
            "        g = qs.shape[1] // ks_.shape[1]\n"
            "        ks_, vs = ks_.repeat_interleave(g, 1), vs.repeat_interleave(g, 1)\n"
            "        Lq, Lk = qs.shape[0], ks_.shape[0]\n"
            "        s = torch.einsum('qhd,khd->hqk', qs, ks_) * scale\n"
            "        pos = torch.arange(Lk - Lq, Lk)[:, None]\n"
            "        key = torch.arange(Lk)[None, :]\n"
            "        mask = key > pos\n"
            "        if window:\n"
            "            mask |= key <= pos - window\n"
            "        s = s.masked_fill(mask, float('-inf'))\n"
            "        out[cu_q[b]:cu_q[b + 1]] = torch.einsum('hqk,khd->qhd', s.softmax(-1), vs)\n"
            "    return out.to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": ["aiter flash_attn_varlen_func (CK fmha)", "triton unified_attention"],
            "sglang": ["aiter flash_attn_varlen / mha_batch_prefill", "triton extend_attention"],
        },
        "tuning_keys": ["batch", "max_q", "q", "heads", "kv_heads", "qk_dim", "v_dim", "max_kv"],
    },
    "v4_attention": {
        "computes": "DeepSeek-V4 attention on one layer type: each query attends its "
        "local window (window keys of the uncompressed KV) plus a compressed pool of "
        "pool_kv entries (one per compress tokens; on CSA layers, compress 4, the "
        "indexer's top-k of them). One shared KV head of head_dim = qk_dim. Decode "
        "rows are per-sequence q; prefill rows carry the step's total q.",
        "inputs": [
            {"name": "q", "shape": ["batch", "q", "heads", "qk_dim"], "dtype": "bf16"},
            {
                "name": "window_kv",
                "shape": ["batch", "min(max_kv, window)", 1, "qk_dim"],
                "dtype": "$dtype",
            },
            {"name": "pool_kv", "shape": ["batch", "pool_kv", 1, "qk_dim"], "dtype": "$dtype"},
            {
                "name": "pool_index",
                "shape": ["batch", "q", "pool_kv"],
                "dtype": "int32",
                "note": "CSA only: the indexer's top-k selection",
            },
        ],
        "outputs": [{"name": "out", "shape": ["batch", "q", "heads", "v_dim"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["DeepSeek-V4 sparse attention kernels (FlashMLA-style)"],
            "sglang": ["DeepSeek-V4 NSA backend"],
        },
        "tuning_keys": ["batch", "q", "heads", "qk_dim", "window", "compress", "pool_kv"],
    },
    "indexer_score": {
        "computes": "Sparse-attention indexer scoring: for each query token, "
        "score[j] = sum_h w[h] * relu(q[h] . k[j]) over the max_kv candidate keys "
        "in the index cache (on DeepSeek-V4 CSA layers the candidates are the "
        "compressed pool).",
        "inputs": [
            {"name": "q", "shape": ["batch * q", "heads", "dim"], "dtype": "$dtype"},
            {
                "name": "k_cache",
                "shape": ["batch", "max_kv", "dim"],
                "dtype": "$dtype",
                "layout": "paged",
            },
            {"name": "weights", "shape": ["batch * q", "heads"], "dtype": "fp32"},
        ],
        "outputs": [{"name": "scores", "shape": ["batch * q", "max_kv"], "dtype": "fp32"}],
        "reference": (
            "def ref(q, k, w):\n"
            "    # q: [T, H, D], k: [T, L, D] (each token's candidates), w: [T, H]\n"
            "    s = torch.einsum('thd,tld->thl', q.float(), k.float()).relu()\n"
            "    return (s * w.float()[..., None]).sum(1)"
        ),
        "engines": {
            "vllm": ["deep_gemm fp8_paged_mqa_logits / triton indexer"],
            "sglang": ["NSA indexer (fp8 mqa logits)"],
        },
        "tuning_keys": ["batch", "q", "heads", "dim", "max_kv"],
    },
    "indexer_topk": {
        "computes": "Top-k over each query row's candidate scores.",
        "inputs": [{"name": "scores", "shape": ["rows", "candidates"], "dtype": "fp32"}],
        "outputs": [{"name": "indices", "shape": ["rows", "k"], "dtype": "int32"}],
        "reference": "def ref(scores, k):\n    return scores.topk(k, dim=-1).indices.int()",
        "engines": {"vllm": ["fast_topk / radix top-k"], "sglang": ["fast_topk"]},
        "tuning_keys": ["rows", "candidates", "k"],
    },
    "index_cache_write": {
        "computes": "Writes each new token's index key (quantized) into the index cache.",
        "inputs": [{"name": "k", "shape": ["tokens", "dim"], "dtype": "bf16"}],
        "outputs": [{"name": "index_cache", "shape": ["slots", "dim"], "dtype": "$dtype"}],
        "reference": None,
        "engines": {"vllm": ["indexer_k_quant_and_cache"], "sglang": ["index cache store"]},
        "tuning_keys": ["tokens", "dim"],
    },
    "linear_attention_prefill": {
        "computes": "Chunked linear-attention prefill (GDN gated delta rule or KDA): "
        "q tokens over batch sequences (max_q the longest) update each sequence's "
        "[value_heads, dk, dv] fp32 state and emit outputs.",
        "inputs": [
            {"name": "q", "shape": ["q", "key_heads", "dk"], "dtype": "bf16"},
            {"name": "k", "shape": ["q", "key_heads", "dk"], "dtype": "bf16"},
            {"name": "v", "shape": ["q", "value_heads", "dv"], "dtype": "bf16"},
            {
                "name": "g",
                "shape": ["q", "value_heads"],
                "dtype": "fp32",
                "note": "decay (per key dim for KDA)",
            },
            {"name": "beta", "shape": ["q", "value_heads"], "dtype": "fp32"},
            {"name": "state", "shape": ["batch", "value_heads", "dk", "dv"], "dtype": "fp32"},
        ],
        "outputs": [{"name": "o", "shape": ["q", "value_heads", "dv"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["fla chunk_gated_delta_rule / chunk_kda (triton)"],
            "sglang": ["fla chunk_gated_delta_rule (triton)"],
        },
        "tuning_keys": ["batch", "max_q", "value_heads", "dk", "dv"],
    },
    "linear_attention_decode": {
        "computes": "Recurrent linear-attention decode: each of batch sequences applies "
        "its q tokens to its fp32 state.",
        "inputs": [
            {"name": "q", "shape": ["batch", "q", "key_heads", "dk"], "dtype": "bf16"},
            {"name": "k", "shape": ["batch", "q", "key_heads", "dk"], "dtype": "bf16"},
            {"name": "v", "shape": ["batch", "q", "value_heads", "dv"], "dtype": "bf16"},
            {"name": "state", "shape": ["batch", "value_heads", "dk", "dv"], "dtype": "fp32"},
        ],
        "outputs": [{"name": "o", "shape": ["batch", "q", "value_heads", "dv"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["fla fused_recurrent_gated_delta_rule / fused_recurrent_kda"],
            "sglang": ["fla fused_recurrent_gated_delta_rule"],
        },
        "tuning_keys": ["batch", "q", "value_heads", "dk", "dv"],
    },
    "causal_conv1d": {
        "computes": "Depthwise causal conv1d over the step's tokens with per-sequence conv state.",
        "inputs": [
            {"name": "x", "shape": ["tokens", "channels"], "dtype": "bf16"},
            {"name": "weight", "shape": ["channels", "kernel"], "dtype": "bf16"},
        ],
        "outputs": [{"name": "y", "shape": ["tokens", "channels"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["causal_conv1d_fn / causal_conv1d_update (triton)"],
            "sglang": ["causal_conv1d (triton)"],
        },
        "tuning_keys": ["tokens", "channels", "kernel"],
    },
    "rmsnorm": {
        "computes": "RMSNorm, fused with the residual add: r = x + residual; "
        "y = r * rsqrt(mean(r^2) + eps) * weight.",
        "inputs": [
            {"name": "x", "shape": ["tokens", "width"], "dtype": "bf16"},
            {"name": "residual", "shape": ["tokens", "width"], "dtype": "bf16"},
            {"name": "weight", "shape": ["width"], "dtype": "bf16"},
        ],
        "outputs": [
            {"name": "y", "shape": ["tokens", "width"], "dtype": "bf16"},
            {"name": "residual_out", "shape": ["tokens", "width"], "dtype": "bf16"},
        ],
        "reference": (
            "def ref(x, residual, weight, eps=1e-6):\n"
            "    r = x.float() + residual.float()\n"
            "    y = r * torch.rsqrt(r.pow(2).mean(-1, keepdim=True) + eps) * weight.float()\n"
            "    return y.to(torch.bfloat16), r.to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": ["aiter rmsnorm2d_fwd_with_add", "fused_add_rms_norm"],
            "sglang": ["aiter rmsnorm2d_fwd_with_add", "fused_add_rmsnorm"],
        },
        "tuning_keys": ["tokens", "width"],
    },
    "qk_rmsnorm": {
        "computes": "Per-head RMSNorm of the queries and keys (QK-norm) over head_dim.",
        "inputs": [{"name": "qk", "shape": ["tokens", "heads", "width"], "dtype": "bf16"}],
        "outputs": [{"name": "qk", "shape": ["tokens", "heads", "width"], "dtype": "bf16"}],
        "reference": (
            "def ref(x, weight, eps=1e-6):\n"
            "    xf = x.float()\n"
            "    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": ["rms_norm per head (often fused with RoPE)"],
            "sglang": ["rmsnorm per head"],
        },
        "tuning_keys": ["tokens", "heads", "width"],
    },
    "gated_rmsnorm": {
        "computes": "The linear-attention output norm: per head RMSNorm, then times silu(gate).",
        "inputs": [
            {"name": "o", "shape": ["tokens", "heads", "width"], "dtype": "bf16"},
            {"name": "gate", "shape": ["tokens", "heads", "width"], "dtype": "bf16"},
        ],
        "outputs": [{"name": "y", "shape": ["tokens", "heads", "width"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["fla fused gated rmsnorm (triton)"],
            "sglang": ["gated rmsnorm (triton)"],
        },
        "tuning_keys": ["tokens", "heads", "width"],
    },
    "act_and_mul": {
        "computes": "Gated activation: y = act(x[:, :width]) * x[:, width:] (SiLU for SwiGLU).",
        "inputs": [{"name": "x", "shape": ["$rows", "2 * width"], "dtype": "bf16"}],
        "outputs": [{"name": "y", "shape": ["$rows", "width"], "dtype": "$out"}],
        "reference": (
            "def ref(x):\n"
            "    a, b = x.float().chunk(2, dim=-1)\n"
            "    return (torch.nn.functional.silu(a) * b).to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": ["aiter silu_and_mul", "silu_and_mul (fused with fp8 quant when quantized)"],
            "sglang": ["aiter silu_and_mul", "silu_and_mul"],
        },
        "tuning_keys": ["$rows", "width"],
    },
    "rope": {
        "computes": "Rotary position embedding applied in place to the queries and keys "
        "(dim rotary channels per head) at each token's position.",
        "inputs": [
            {"name": "positions", "shape": ["tokens"], "dtype": "int64"},
            {"name": "q", "shape": ["tokens", "heads", "dim"], "dtype": "bf16"},
            {"name": "k", "shape": ["tokens", "kv_heads", "dim"], "dtype": "bf16"},
            {"name": "cos_sin_cache", "shape": ["max_position", "dim"], "dtype": "bf16"},
        ],
        "outputs": [{"name": "q, k", "shape": ["(in place)"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["rotary_embedding (custom op)", "aiter rope_cached"],
            "sglang": ["rotary_embedding", "aiter rope"],
        },
        "tuning_keys": ["tokens", "heads", "kv_heads", "dim"],
    },
    "attn_decode_reduce": {
        "computes": "The second pass of split-KV decode attention: merges each head's "
        "partial outputs over partitions KV partitions, rescaled by their max logits and "
        "exp sums, into one output row.",
        "inputs": [
            {
                "name": "tmp_out",
                "shape": ["batch", "heads", "partitions", "v_dim"],
                "dtype": "bf16",
            },
            {"name": "max_logits", "shape": ["batch", "heads", "partitions"], "dtype": "fp32"},
            {"name": "exp_sums", "shape": ["batch", "heads", "partitions"], "dtype": "fp32"},
        ],
        "outputs": [{"name": "out", "shape": ["batch", "heads", "v_dim"], "dtype": "bf16"}],
        "reference": (
            "def ref(tmp_out, max_logits, exp_sums):\n"
            "    m = max_logits.max(-1, keepdim=True).values\n"
            "    w = exp_sums * torch.exp(max_logits - m)\n"
            "    w = w / w.sum(-1, keepdim=True)\n"
            "    return (tmp_out.float() * w[..., None]).sum(-2).to(torch.bfloat16)"
        ),
        "engines": {
            "vllm": ["paged_attention_ll4mi_reduce_kernel (ROCM_ATTN)", "triton reduce_segments"],
            "sglang": ["triton decode_attention stage 2", "triton reduce_segments"],
        },
        "tuning_keys": ["batch", "heads", "partitions", "v_dim"],
    },
    "moe_pad": {
        "computes": "Copies the MoE input from width to padded channels, zero-filling the "
        "rest, for expert kernels that need padded weights.",
        "inputs": [{"name": "x", "shape": ["tokens", "width"], "dtype": "bf16"}],
        "outputs": [{"name": "x_padded", "shape": ["tokens", "padded"], "dtype": "bf16"}],
        "reference": (
            "def ref(x, padded):\n"
            "    return torch.nn.functional.pad(x, (0, padded - x.shape[-1]))"
        ),
        "engines": {"sglang": ["torch.nn.functional.pad (fill + copy)"]},
        "tuning_keys": ["tokens", "width", "padded"],
    },
    "kv_cache_write": {
        "computes": "Writes each new token's keys and values (dim channels over kv_heads; "
        "K and V together for GQA, the latent for MLA, a pool entry for DeepSeek-V4) into "
        "the paged cache at its slot, quantizing to the cache dtype.",
        "inputs": [
            {"name": "kv", "shape": ["tokens", "kv_heads", "dim"], "dtype": "bf16"},
            {"name": "slot_mapping", "shape": ["tokens"], "dtype": "int64"},
        ],
        "outputs": [
            {
                "name": "kv_cache",
                "shape": ["num_blocks", "$block_size", "kv_heads", "dim"],
                "dtype": "$dtype",
            }
        ],
        "reference": None,
        "engines": {
            "vllm": ["reshape_and_cache_flash", "concat_and_cache_mla (MLA)"],
            "sglang": ["set_kv_buffer / store_cache", "set_mla_kv_buffer (MLA)"],
        },
        "tuning_keys": ["tokens", "kv_heads", "dim", "dtype"],
    },
    "embedding": {
        "computes": "Vocab-parallel embedding lookup over this rank's vocab shard (the "
        "partial rows are then all-reduced over TP).",
        "inputs": [
            {"name": "ids", "shape": ["tokens"], "dtype": "int64"},
            {"name": "table", "shape": ["vocab", "width"], "dtype": "bf16"},
        ],
        "outputs": [{"name": "h", "shape": ["tokens", "width"], "dtype": "bf16"}],
        "reference": None,
        "engines": {
            "vllm": ["VocabParallelEmbedding (torch)"],
            "sglang": ["VocabParallelEmbedding"],
        },
        "tuning_keys": ["tokens", "width"],
    },
    "moe_topk": {
        "computes": "MoE router selection: per token, the topk experts of its router "
        "logits and their weights (softmax for most models; sigmoid with grouped, "
        "bias-corrected selection for DeepSeek-style routers; see params).",
        "inputs": [{"name": "logits", "shape": ["tokens", "experts"], "dtype": "fp32"}],
        "outputs": [
            {"name": "topk_weights", "shape": ["tokens", "topk"], "dtype": "fp32"},
            {"name": "topk_ids", "shape": ["tokens", "topk"], "dtype": "int32"},
        ],
        "reference": (
            "def ref(logits, topk):\n"
            "    w, i = logits.float().softmax(-1).topk(topk, dim=-1)\n"
            "    return w / w.sum(-1, keepdim=True), i.int()"
        ),
        "engines": {
            "vllm": ["topk_softmax", "aiter biased_grouped_topk (DeepSeek)"],
            "sglang": ["topk_softmax", "biased_grouped_topk"],
        },
        "tuning_keys": ["tokens", "experts", "topk"],
    },
    "moe_align": {
        "computes": "Sorts the routed (token, expert) pairs by local expert and pads each "
        "expert's rows to the fused-MoE block size.",
        "inputs": [{"name": "topk_ids", "shape": ["tokens", "topk"], "dtype": "int32"}],
        "outputs": [
            {"name": "sorted_token_ids", "shape": ["tokens * topk + pad"], "dtype": "int32"},
            {"name": "expert_ids", "shape": ["blocks"], "dtype": "int32"},
        ],
        "reference": None,
        "engines": {
            "vllm": ["moe_align_block_size", "aiter moe_sorting"],
            "sglang": ["moe_align_block_size"],
        },
        "tuning_keys": ["tokens", "topk", "local_experts"],
    },
    "moe_sum": {
        "computes": "Weighted reduction of each token's topk expert outputs.",
        "inputs": [
            {"name": "x", "shape": ["tokens", "topk", "width"], "dtype": "bf16"},
            {"name": "topk_weights", "shape": ["tokens", "topk"], "dtype": "fp32"},
        ],
        "outputs": [{"name": "y", "shape": ["tokens", "width"], "dtype": "bf16"}],
        "reference": (
            "def ref(x, w):\n"
            "    return (x.float() * w.float()[..., None]).sum(1).to(torch.bfloat16)"
        ),
        "engines": {"vllm": ["moe_sum (inside fused_moe)"], "sglang": ["moe_sum_reduce"]},
        "tuning_keys": ["tokens", "topk", "width"],
    },
    "act_quant": {
        "computes": "Dynamic activation quantization in front of a low-precision GEMM: "
        "rows x width bf16 to the GEMM's input dtype with its scales (per token, or per "
        "128-wide block for block-scaled fp8 and per 32 for MX formats).",
        "inputs": [{"name": "x", "shape": ["rows", "width"], "dtype": "bf16"}],
        "outputs": [
            {"name": "xq", "shape": ["rows", "width"], "dtype": "$dtype"},
            {"name": "scale", "shape": ["rows", "width / group"], "dtype": "fp32"},
        ],
        "reference": (
            "def ref(x, group=128, fmax=448.0):\n"
            "    t, w = x.shape\n"
            "    xg = x.float().view(t, w // group, group)\n"
            "    s = xg.abs().amax(-1, keepdim=True).clamp(min=1e-12) / fmax\n"
            "    return (xg / s).to(torch.float8_e4m3fn).view(t, w), s.squeeze(-1)"
        ),
        "engines": {
            "vllm": [
                "aiter per_token_quant / dynamic_per_token_scaled_fp8_quant",
                "per_token_group_quant_fp8",
            ],
            "sglang": ["sglang_per_token_group_quant_fp8", "aiter dynamic_mxfp4_quant"],
        },
        "tuning_keys": ["rows", "width", "dtype"],
    },
    "sampling": {
        "computes": "Token sampling over the full vocabulary for each sampled row "
        "(temperature, top-k / top-p, then a draw).",
        "inputs": [{"name": "logits", "shape": ["rows", "vocab"], "dtype": "fp32"}],
        "outputs": [{"name": "tokens", "shape": ["rows"], "dtype": "int64"}],
        "reference": None,
        "engines": {
            "vllm": ["Sampler (torch / aiter top-k top-p)"],
            "sglang": ["Sampler (top_k_top_p)"],
        },
        "tuning_keys": ["rows", "vocab"],
    },
    "collective": {
        "computes": "A collective over ranks GPUs carrying bytes per rank. The message size "
        "decides the algorithm (custom all-reduce, quick-reduce, RCCL, or an EP "
        "dispatch/combine library).",
        "inputs": [{"name": "buffer", "shape": ["bytes / $elem_bytes"], "dtype": "$dtype"}],
        "outputs": [{"name": "buffer", "shape": ["bytes / $elem_bytes"], "dtype": "$dtype"}],
        "reference": None,
        "engines": {
            "vllm": ["custom_all_reduce / quick_reduce / RCCL", "mori / DeepEP (EP a2a)"],
            "sglang": ["custom_all_reduce / RCCL", "mori / DeepEP"],
        },
        "tuning_keys": ["op", "ranks", "bytes"],
    },
}

_GENERIC = {
    "computes": "",
    "inputs": [],
    "outputs": [],
    "reference": None,
    "engines": {},
    "tuning_keys": [],
}

_OP_NOTES = {
    "attn_gate_mul": "Attention output gate: out *= sigmoid(gate), over tokens x width.",
    "hc_sinkhorn": "DeepSeek-V4 mHC: Sinkhorn normalization of each token's streams x streams "
    "mixing matrix, iters iterations, fp32.",
    "hc_apply": "DeepSeek-V4 mHC: mixes each token's streams residual streams of width channels.",
    "attn_res_mix": "Kimi-K3 attention residuals: a weighted mix of blocks earlier block outputs.",
    "pool_cache_write": "DeepSeek-V4: writes compressed-pool entries (one per compress tokens).",
    "moe_latent_rmsnorm": "Latent MoE: RMSNorm of the latent the experts run on.",
    "embedding": None,
    "q_a_rmsnorm": "MLA / V4: RMSNorm of the query LoRA latent.",
    "kv_a_rmsnorm": "MLA: RMSNorm of the KV latent before caching.",
    "kv_rmsnorm": "DeepSeek-V4: RMSNorm of the shared KV head.",
    "tp_allreduce": "TP all-reduce after the attention and MLP output projections.",
    "tp_allgather": "Attention-DP: gathers the ranks' tokens for the TP MLP.",
    "tp_reduce_scatter": "Attention-DP: scatters the MLP output back to the owning ranks.",
    "ep_dispatch": "EP all-to-all sending each routed token to its expert's rank.",
    "ep_combine": "EP all-to-all returning expert outputs to the token's rank.",
    "embed_allreduce": "All-reduce of the vocab-parallel embedding's partial rows.",
    "logits_allgather": "Gathers the vocab-parallel logits of the sampled rows.",
    "pp_send_recv": "Pipeline handoff of the step's hidden states to the next stage.",
    "dcp_allgather": "Decode context parallelism: gathers the heads' partial attention.",
    "dcp_reduce_scatter": "Decode context parallelism: scatters the combined output.",
    "draft_eh_proj": "MTP drafter: projects the concatenated [embedding, hidden] to hidden.",
    "draft_fc": "EAGLE3 / DFlash drafter: fuses the target's captured hidden states.",
    "draft_serial_head": "DFlash serial head: a low-rank vocabulary projection per draft position.",
    "lm_head": "The LM head over the sampled rows, on this rank's vocab shard.",
    "moe_router": "The MoE router logits (replicated over TP).",
}


_BASIS = {
    "origami": "Origami's tile-level GEMM model of the target GPU",
    "sdpa": "the SDPA simulator's tile-level attention model",
    "collectives": "the projection's collective model (algorithm and link latency/bandwidth)",
    "roofline": "a roofline at 56.6% of peak HBM bandwidth and 70% of peak compute, so its "
    "headroom over speed of light is that assumption, not a finding: measure it first",
    "measured": "a measured time",
}


# ── speed of light ──────────────────────────────────────────────────────────


def speed_of_light_us(kind: str, op: str, dtype: str, dims: dict, bw_gbps: float, tflops: float):
    """``(us, flops, bytes, bound)`` at the GPU's peak bandwidth and compute."""
    k = ks.Kernel(kind, op, dtype, tuple(dims.items()))
    try:
        flops, nbytes = ks._traffic(k, dims)
    except (KeyError, TypeError, ZeroDivisionError):
        return None, None, None, None
    if not flops and not nbytes:
        return None, flops, nbytes, None
    peak = tflops * (_COMPUTE_X.get(dtype, 1.0) if kind == "gemm" else 1.0)
    t_mem = nbytes / (bw_gbps * 1e9)
    t_cmp = flops / (peak * 1e12) if peak else 0.0
    bound = "compute" if t_cmp > t_mem else "memory"
    return 1e6 * max(t_mem, t_cmp), flops, nbytes, bound


# ── the deployment ──────────────────────────────────────────────────────────


def deployment(cfg, args=None, timer=None) -> dict:
    """The deployment the tasks were derived for, as an agent needs it."""
    mc, req = cfg.model_config, cfg.request_config
    tp, atp, dp, ep, dcp = ks._ranks(cfg.model_parallel_config)
    g = lambda o, f: getattr(o, f, None)  # noqa: E731
    arch = {
        f: g(mc, f)
        for f in (
            "num_layers",
            "hidden_size",
            "num_attention_heads",
            "num_query_groups",
            "kv_channels",
            "multi_latent_attention",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_head_dim",
            "qk_pos_emb_head_dim",
            "v_head_dim",
            "ffn_hidden_size",
            "num_experts",
            "moe_router_topk",
            "moe_ffn_hidden_size",
            "moe_shared_expert_intermediate_size",
            "moe_latent_hidden_size",
            "moe_router_num_groups",
            "moe_router_group_topk",
            "moe_router_topk_scaling_factor",
            "padded_vocab_size",
            "swiglu",
            "qk_layernorm",
            "attention_output_gate",
            "sink_sliding_window",
            "attn_sliding_window",
            "linear_attention_kind",
            "linear_attention_layers",
            "index_topk",
            "sparse_index_n_heads",
            "o_lora_rank",
            "o_groups",
            "hc_mult",
        )
        if g(mc, f) not in (None, 0, False, "")
    }
    block = int(g(req, "kv_block_size") or 0)
    engine = str(g(req, "serving_engine") or "vllm").lower()
    return {
        "model": g(mc, "model_name") or os.environ.get("INFERASIM_MODEL"),
        "architecture": arch,
        "parallel": {
            "tp": tp,
            "attention_tp": atp,
            "attention_dp": dp,
            "ep": ep,
            "dcp": dcp,
            "pp": int(g(cfg.model_parallel_config, "pipeline_model_parallel_size") or 1),
        },
        "dtypes": {
            "linear_weights": gemm_shapes._weight_dtype(cfg, "linear_weight_dtype"),
            "moe_experts": gemm_shapes._weight_dtype(cfg, "moe_expert_dtype"),
            "activations": ks._act_quant_dtype(cfg) or "bf16",
            "kv_cache": ks._kv_dtype(cfg),
            "fp8_recipe": g(mc, "fp8"),
        },
        "engine": engine,
        "attention_backend": g(req, "attention_backend"),
        "cudagraph_mode": g(req, "cudagraph_mode"),
        "cudagraph_capture_sizes": list(ks.capture_sizes(cfg)),
        "kv_block_size": block or (16 if engine == "vllm" else 1),
        "kv_block_size_assumed": not block,
        "gpu_arch": g(args, "gpu_arch") if args is not None else None,
        "peak_hbm_gbps": timer.bw if timer is not None else None,
        "peak_tflops_bf16": timer.peak if timer is not None else None,
    }


# ── tasks ───────────────────────────────────────────────────────────────────


_META = {
    "pool",
    "stage",
    "kind",
    "op",
    "dtype",
    "calls",
    "est_us",
    "est_total_ms",
    "est_source",
    "est_us_base",
    "speedup",
}


def _gpus(pool: str) -> str:
    """The GPUs a pool runs on: a drafter shares its target's."""
    return pool.split("/", 1)[0]


def _tolerance(kind: str, dtype: str, outputs: list) -> dict:
    """What a rewrite's output must match the reference to: by the output's precision."""
    if kind == "quant":
        return _TOLERANCE.get(dtype, _TOLERANCE["fp8"])
    out = str(outputs[0].get("dtype")) if outputs else "bf16"
    if out.startswith("$"):
        out = dtype
    if kind == "gemm" and dtype not in ("bf16", "fp16", "fp32"):
        # A low-precision GEMM is checked against the bf16 product of its dequantized operands.
        return {
            **_TOLERANCE["bf16"],
            "note": "against the bf16 product of the dequantized operands",
        }
    return _TOLERANCE.get(out, _TOLERANCE["bf16"])


_OP_SIGNATURES = {
    "moe_down": {
        "inputs": [
            {
                "name": "x",
                "shape": ["m", "topk", "k"],
                "dtype": "$act",
                "note": "the activation of each routed row",
            },
            {"name": "w", "shape": ["groups", "n", "k"], "dtype": "$dtype", "layout": "NT"},
            {"name": "topk_ids", "shape": ["m", "topk"], "dtype": "int32"},
        ],
        "outputs": [
            {
                "name": "out",
                "shape": ["m", "topk", "n"],
                "dtype": "bf16",
                "note": "reduced over topk by moe_sum",
            }
        ],
    },
}


_VAR = re.compile(r"\$([a-z_]+)")


def _subst(shape, ctx: dict):
    out = []
    for s in shape:
        if isinstance(s, str) and s.startswith("$") and s[1:] in ctx:
            out.append(ctx[s[1:]])
        elif isinstance(s, str):
            out.append(_VAR.sub(lambda m: str(ctx.get(m.group(1), m.group(0))), s))
        else:
            out.append(s)
    return out


_FUNCS = {"ceil": math.ceil, "min": min, "max": max}
_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
}


def _eval(node, names: dict):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and isinstance(names.get(node.id), (int, float)):
        return names[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left, names), _eval(node.right, names))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*(_eval(a, names) for a in node.args))
    raise KeyError(ast.dump(node))


def _dim(v, names: dict):
    """One tensor dimension for a case: a number when every name in it is known."""
    if not isinstance(v, str):
        return v
    try:
        x = _eval(ast.parse(v.replace("sum(kv)", "(batch * mean_kv)"), mode="eval").body, names)
    except (KeyError, SyntaxError, TypeError, ZeroDivisionError):
        return v
    return int(math.ceil(x)) if isinstance(x, float) else int(x)


def _tensors(spec_tensors: list, dims: dict, ctx: dict) -> list:
    names = {**{k: v for k, v in ctx.items() if isinstance(v, (int, float))}, **dims}
    return [
        {"name": t["name"], "shape": [_dim(v, names) for v in t["shape"]], "dtype": t.get("dtype")}
        for t in spec_tensors
    ]


def _signature(spec: dict, dtype: str, ctx: dict) -> tuple[list, list]:
    def one(t):
        t = dict(t)
        t["shape"] = _subst(t["shape"], ctx)
        if isinstance(t.get("dtype"), str) and t["dtype"].startswith("$"):
            t["dtype"] = ctx.get(t["dtype"][1:], dtype)
        return t

    return [one(t) for t in spec["inputs"]], [one(t) for t in spec["outputs"]]


_HINT_PRECISION = {"bf16": ("bf16", "fp16", "fp32"), "fp8": ("fp8",), "fp4": ("fp4", "mxfp4", "nvfp4", "int4")}


def _engine_hints(engines: dict[str, list[str]], engine: str, dtype: str, mla: bool) -> list[str]:
    """The engine's kernels for an op, without those tagged for another precision or attention type."""
    out = []
    for hint in engines.get(engine) or next(iter(engines.values()), []):
        tag = hint[hint.rfind("(") + 1 : -1].lower().split() if hint.endswith(")") else []
        if "mla" in tag and not mla:
            continue
        precision = next((p for p in _HINT_PRECISION if p in tag), None)
        if precision and str(dtype).lower() not in _HINT_PRECISION[precision]:
            continue
        out.append(hint)
    return out


def build(rows: list[dict], cfg, *, timer=None, args=None, provenance=None, e2e=None) -> dict:
    """Kernel tasks from :meth:`kernel_shapes.KernelTrace.kernels` rows."""
    bw = timer.bw if timer is not None else 5300.0
    tflops = timer.peak if timer is not None else 1307.0
    dep = deployment(cfg, args, timer)
    act = dep["dtypes"]["activations"]
    mla = bool(getattr(cfg.model_config, "multi_latent_attention", False))
    totals: dict[str, float] = defaultdict(float)
    gpus: dict[str, float] = defaultdict(float)
    for r in rows:
        totals[r["pool"]] += r.get("est_total_ms") or 0.0
        gpus[_gpus(r["pool"])] += r.get("est_total_ms") or 0.0
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["pool"], r["kind"], r["op"], r["dtype"])].append(r)
    tasks = []
    for (pool, kind, op, dtype), rs in groups.items():
        dims0 = {k: v for k, v in rs[0].items() if k not in _META}
        fam = _family(kind, op, dims0)
        spec = {**_FAMILIES.get(fam, _GENERIC), **_OP_SIGNATURES.get(op, {})}
        ctx = {
            "dtype": dtype,
            "act": act if kind == "gemm" and dtype not in ("bf16", "fp16", "fp32") else "bf16",
            "block_size": dep["kv_block_size"],
            "rows": "rows" if "rows" in dims0 else "tokens",
            "out": "bf16",
            "elem_bytes": _ELEM_BYTES.get(dtype, 2),
        }
        inputs, outputs = _signature(spec, dtype, ctx)
        cases = []
        total_ms = sum(r.get("est_total_ms") or 0.0 for r in rs)
        total_calls = sum(r["calls"] for r in rs)
        sol_ms = 0.0
        merged: dict[tuple, dict] = {}
        for r in rs:
            dims = {k: v for k, v in r.items() if k not in _META}
            key = tuple(sorted(dims.items()))
            m = merged.setdefault(
                key, {"dims": dims, "calls": 0, "est_total_ms": 0.0, "stages": set(), "row": r}
            )
            m["calls"] += r["calls"]
            m["est_total_ms"] += r.get("est_total_ms") or 0.0
            m["stages"].add(r.get("stage", 0))
        for m in sorted(merged.values(), key=lambda m: (-m["est_total_ms"], -m["calls"])):
            r = m["row"]
            us, flops, nbytes, bound = speed_of_light_us(kind, op, dtype, m["dims"], bw, tflops)
            if us is not None:
                sol_ms += us * m["calls"] / 1000.0
            case = {
                "dims": m["dims"],
                "calls": m["calls"],
                "call_share_pct": round(100.0 * m["calls"] / total_calls, 3)
                if total_calls
                else None,
                "est_us": r.get("est_us"),
                "est_total_ms": round(m["est_total_ms"], 3),
                "time_share_pct": round(100.0 * m["est_total_ms"] / total_ms, 3)
                if total_ms
                else None,
                "est_source": r.get("est_source"),
                "speed_of_light_us": None if us is None else round(us, 3),
                "flops": flops,
                "bytes": nbytes,
                "bound": bound,
                "stages": sorted(m["stages"]),
            }
            case["inputs"] = _tensors(inputs, m["dims"], ctx)
            case["outputs"] = _tensors(outputs, m["dims"], ctx)
            if r.get("speedup") is not None:
                case["speedup_applied"] = r["speedup"]
            cases.append(case)
        kept, acc = [], 0.0
        for c in cases:
            if len(kept) >= MAX_CASES or (total_ms and acc >= CASE_COVERAGE * total_ms):
                break
            kept.append(c)
            acc += c["est_total_ms"]
        if not total_ms:
            kept = cases[:MAX_CASES]
        for i, c in enumerate(kept, 1):
            c["case_id"] = f"c{i:02d}"
        pool_ms = gpus[_gpus(pool)] or 0.0
        share = 100.0 * total_ms / pool_ms if pool_ms else None
        gain = None
        if share is not None and total_ms and sol_ms:
            gain = share * max(0.0, 1.0 - sol_ms / total_ms)
        sources: dict[str, float] = defaultdict(float)
        for r in rs:
            sources[str(r.get("est_source"))] += r.get("est_total_ms") or 0.0
        main_src = max(sources, key=sources.get) if sources else None
        basis = _BASIS.get(str(main_src).split("+")[0], None)
        warning = None
        if sol_ms and total_ms < sol_ms:
            warning = (
                "the estimate is under speed of light for these shapes: its time source"
                + (" or the speedup applied" if any(r.get("speedup") for r in rs) else "")
                + " is too optimistic"
            )
        ranges = {}
        for c in cases:
            for d, v in c["dims"].items():
                if isinstance(v, (int, float)):
                    lo, hi = ranges.get(d, (v, v))
                    ranges[d] = (min(lo, v), max(hi, v))
        tasks.append(
            {
                "pool": pool,
                "kind": kind,
                "op": op,
                "dtype": dtype,
                "family": fam,
                "computes": spec["computes"] or _OP_NOTES.get(op) or "",
                "note": _OP_NOTES.get(op) if spec["computes"] else None,
                "inputs": inputs,
                "outputs": outputs,
                "reference": spec["reference"],
                "tolerance": _tolerance(kind, dtype, outputs),
                "engine_kernels": _engine_hints(spec["engines"], dep["engine"], dtype, mla),
                "tuning_keys": [
                    ctx.get(k[1:], k) if k.startswith("$") else k for k in spec["tuning_keys"]
                ],
                "calls": total_calls,
                "distinct_shapes": len(cases),
                "est_total_ms": round(total_ms, 3),
                "share_pct": None if share is None else round(share, 3),
                "speed_of_light_ms": round(sol_ms, 3) if sol_ms else None,
                "headroom_x": round(total_ms / sol_ms, 3) if sol_ms and total_ms else None,
                "est_sources_ms": {k: round(v, 3) for k, v in sources.items()},
                "estimate_basis": basis,
                "warning": warning,
                "e2e_gain_pct_at_speed_of_light": None if gain is None else round(gain, 3),
                "dims_range": {d: list(v) for d, v in sorted(ranges.items())},
                "cases_coverage_pct": round(100.0 * acc / total_ms, 2) if total_ms else None,
                "cases": kept,
            }
        )
    tasks.sort(key=lambda t: (-(t["est_total_ms"] or 0.0), -t["calls"], t["pool"], t["op"]))
    for i, t in enumerate(tasks, 1):
        t["task_id"] = f"t{i:03d}"
        t["priority"] = i
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "provenance": provenance or {"source": "des"},
        "deployment": dep,
        "totals": {
            "est_gpu_ms": {p: round(v, 3) for p, v in totals.items()},
            "share_basis": "share_pct is of the kernel time on the task's GPUs: its pool's, "
            "plus its drafter's (a drafter's tasks share the target's).",
            "kernels": len(rows),
            "tasks": len(tasks),
            "timing": "est_us is one call on one rank. Totals add calls times est_us and "
            "ignore overlap between kernels, so shares are of GPU kernel time, not wall time.",
        },
        "e2e": e2e,
        "tasks": tasks,
    }


def write(doc: dict, path: str) -> None:
    """``path`` gets the JSON; the same path with ``.md`` the Markdown rendering."""
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1)
    md = path[:-5] + ".md" if path.endswith(".json") else path + ".md"
    with open(md, "w") as fh:
        fh.write(markdown(doc))


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def markdown(doc: dict, max_tasks: int = 30, max_cases: int = 8) -> str:
    dep, tasks = doc["deployment"], doc["tasks"]
    prov = doc.get("provenance") or {}
    out = ["# Kernel tasks", ""]
    out.append(
        f"Model `{dep.get('model')}` on {dep.get('gpu_arch') or 'the target GPU'}, "
        f"{dep['engine']}, TP {dep['parallel']['tp']}, EP {dep['parallel']['ep']}, "
        f"PP {dep['parallel']['pp']}, attention-DP {dep['parallel']['attention_dp']}. "
        f"Weights {dep['dtypes']['linear_weights']} (experts {dep['dtypes']['moe_experts']}), "
        f"activations {dep['dtypes']['activations']}, KV cache {dep['dtypes']['kv_cache']}."
    )
    out.append("")
    src = prov.get("source", "des")
    line = f"Derived from a {'saved step mix' if src == 'step_mix' else 'DES replay'}"
    if src == "step_mix":
        line += " (exact reuse)" if prov.get("exact") else " (approximate reuse)"
    out.append(line + ". Shapes are per rank. Times are estimates, not measurements.")
    for r in prov.get("reasons") or []:
        out.append(f"- {r}")
    e2e = doc.get("e2e") or {}
    if e2e.get("des"):
        d = e2e["des"]
        out.append("")
        out.append(
            f"Re-priced end to end: makespan x{_fmt(d.get('makespan_ratio'))}"
            + (
                f", {_fmt(d.get('system_throughput_tps'))} output tok/s"
                if d.get("system_throughput_tps")
                else ""
            )
            + "."
        )
    out += [
        "",
        "## Tasks by estimated GPU time",
        "",
        "| task | pool | op | dtype | share % | est ms | speed-of-light ms | headroom | e2e gain % at SoL | shapes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in tasks[:max_tasks]:
        out.append(
            f"| {t['task_id']} | {t['pool']} | `{t['op']}` | {t['dtype']} | {_fmt(t['share_pct'])} | "
            f"{_fmt(t['est_total_ms'])} | {_fmt(t['speed_of_light_ms'])} | {_fmt(t['headroom_x'])}"
            f"{'*' if str(t.get('estimate_basis') or '').startswith('a roofline') else ''} | "
            f"{_fmt(t['e2e_gain_pct_at_speed_of_light'])} | {t['distinct_shapes']} |"
        )
    out += ["", "\\* the estimate is itself a roofline, so this headroom is an assumption."]
    for t in tasks[:max_tasks]:
        out += ["", f"## {t['task_id']}: `{t['op']}` ({t['pool']}, {t['dtype']})", ""]
        out.append(t["computes"] or "(no description)")
        if t.get("note"):
            out.append("")
            out.append(t["note"])
        if t["inputs"]:
            out.append("")
            for x in t["inputs"]:
                out.append(
                    f"- in `{x['name']}`: [{', '.join(map(str, x['shape']))}] {x.get('dtype', '')}"
                    + (f", {x['layout']}" if x.get("layout") else "")
                )
            for x in t["outputs"]:
                out.append(
                    f"- out `{x['name']}`: [{', '.join(map(str, x['shape']))}] {x.get('dtype', '')}"
                )
        if t.get("warning"):
            out += ["", f"**Warning:** {t['warning']}."]
        if t.get("estimate_basis"):
            out.append("")
            out.append(f"Estimated with {t['estimate_basis']}.")
        if t["engine_kernels"]:
            out.append("")
            out.append("Usually dispatched to: " + "; ".join(t["engine_kernels"]) + ".")
        if t["tuning_keys"]:
            out.append(f"Tuner keys: {', '.join(t['tuning_keys'])}.")
        cases = t["cases"][:max_cases]
        if cases:
            dims = list(cases[0]["dims"])
            out += [
                "",
                "| case | " + " | ".join(dims) + " | calls | time % | est us | SoL us | bound |",
                "|---|" + "---|" * len(dims) + "---|---|---|---|---|",
            ]
            for c in cases:
                out.append(
                    f"| {c['case_id']} | "
                    + " | ".join(_fmt(c["dims"].get(d)) for d in dims)
                    + f" | {c['calls']} | {_fmt(c['time_share_pct'])} | {_fmt(c['est_us'])} | "
                    f"{_fmt(c['speed_of_light_us'])} | {c['bound'] or '-'} |"
                )
        if t["reference"]:
            out += ["", "Reference:", "", "```python", "import torch", "", t["reference"], "```"]
    out.append("")
    return "\n".join(out)
