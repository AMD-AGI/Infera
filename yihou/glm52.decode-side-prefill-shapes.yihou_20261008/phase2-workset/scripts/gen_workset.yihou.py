#!/usr/bin/env python3
"""Generate a FlashInfer-Trace (flashinfer-bench) workset for the GLM-5.2-MXFP4 decode leg.

Standard library only (the login node has no numpy/torch and kills processes above 4 GB);
peak memory is bounded by the largest single blob (~51 MB). Deterministic for a given --seed.

Layout written under --out (FlashInfer-Trace dataset layout):
  definitions/<op_type>/<name>.json
  workloads/<op_type>/<name>.jsonl          one Trace line per workload (solution/evaluation null)
  blob/workloads/<op_type>/<name>/*.safetensors
  glm52_workload_meta.json                  per-uuid phase, bs_graph, probability, calls/iteration

Batch model (see the report): per attention-DP rank B ~ Normal(mu, sigma) rounded and clamped to
[lo, hi]; under DP attention + CUDA graph every rank runs at
bs_graph = snap_up(max over the 4 ranks of B) (DpPaddingMode.MAX_LEN, captured bs list).
"""
import argparse
import json
import math
import os
import random
import struct
import uuid
from collections import Counter
from pathlib import Path

# ---- deployment constants (decode leg, server-info.29257.json + model config.json) ---------
DP = 4                     # attention-DP ranks (TP4 / attn-DP4 / EP1)
DRAFT = 6                  # speculative_num_draft_tokens -> tokens per request in verify/extend
HIDDEN = 6144
N_EXPERTS = 257            # 256 routed + 1 fused shared expert
N_ROUTED = 256
TOPK = 8
MOE_I = 512                # moe_intermediate_size 2048 / TP4
HEADS = 64                 # attention heads per rank (64, attn_tp 1)
KV_LORA, ROPE = 512, 64
IDX_HEADS, IDX_DIM = 32, 128
TOPK_ATTN = 2048
PAGE = 64
POOL_TOKENS = 2_226_368    # decode max_total_num_tokens (per DP rank)
POOL_PAGES = POOL_TOKENS // PAGE
VOCAB_SHARD = 154880 // 4  # vocab-parallel lm_head shard
CAPTURED_BS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 40, 44, 48, 52, 56, 60, 64]
SM_SCALE = 1.0 / math.sqrt(192 + 64)   # (qk_nope 192 + rope 64) ** -0.5 = 0.0625
# Context per running request on the decode leg, time-weighted (decode-0.log of the reference
# deployment, profiling window): p10 118,250 / p50 236,928 / p90 555,093; observed per-rank
# #token max 2,169,216. Log-normal fitted on p50/p90; per-rank total capped at the observed max.
CTX_MEDIAN, CTX_SIGMA, CTX_RANK_CAP = 236_928, math.log(555_093 / 236_928) / 1.2816, 2_169_216

MODEL_TAGS = ["model:glm-5.2-mxfp4", "tp:4", "dp_attention:4", "ep:1", "gpu:mi355x",
              "framework:sglang-0.5.19.dev20260917+ga9fb1c3238", "status:draft"]


# ---- safetensors (stdlib) --------------------------------------------------------------------
def write_safetensors(path, tensors):
    """tensors: {key: (dtype_code, shape, bytes)} with dtype_code in I32/I8/F32."""
    header, offset, payload = {}, 0, []
    for key, (code, shape, data) in tensors.items():
        header[key] = {"dtype": code, "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        payload.append(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * ((8 - len(raw) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for chunk in payload:
            f.write(chunk)


def i32(values):
    return struct.pack(f"<{len(values)}i", *values)


def f32(values):
    return struct.pack(f"<{len(values)}f", *values)


# ---- batch / context model -------------------------------------------------------------------
def snap(b):
    return next(c for c in CAPTURED_BS if c >= b)


def batch_distribution(mu, sigma, lo, hi, iters, rng):
    per, graph = Counter(), Counter()
    for _ in range(iters):
        bs = [min(hi, max(lo, round(rng.gauss(mu, sigma)))) for _ in range(DP)]
        per.update(bs)
        graph[snap(max(bs))] += 1
    return ({k: v / (iters * DP) for k, v in sorted(per.items())},
            {k: v / iters for k, v in sorted(graph.items())})


def sample_contexts(n, rng):
    ctx = [max(4096, min(1_000_000, int(math.exp(rng.gauss(math.log(CTX_MEDIAN), CTX_SIGMA))))) for _ in range(n)]
    total = sum(ctx)
    if total > CTX_RANK_CAP:
        ctx = [max(4096, int(c * CTX_RANK_CAP / total)) for c in ctx]
    return ctx


# ---- definitions -----------------------------------------------------------------------------
def axis_var(desc):
    return {"type": "var", "description": desc}


def axis_const(v, desc):
    return {"type": "const", "value": v, "description": desc}


def T(shape, dtype, desc):
    return {"shape": shape, "dtype": dtype, "description": desc}


REF_MXFP4 = '''
FP4_GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def _fp4_lut(device):
    pos = torch.tensor(FP4_GRID, dtype=torch.float32, device=device)
    return torch.cat([pos, -pos])  # nibble 0..7 positive, 8..15 negative (sign bit 3)


def dequant_mxfp4(packed, scales):
    """packed: int8 [..., K/2] (two E2M1 nibbles per byte, low nibble = even element);
    scales: int8 [..., K/32] holding UE8M0 exponents (value = 2 ** (byte - 127))."""
    b = packed.view(torch.uint8).to(torch.int64)
    lut = _fp4_lut(packed.device)
    vals = torch.stack([lut[b & 0xF], lut[b >> 4]], dim=-1).flatten(-2)
    s = torch.pow(2.0, scales.view(torch.uint8).to(torch.float32) - 127.0)
    return vals * s.repeat_interleave(32, dim=-1)


def quant_dequant_mxfp4(x):
    """Dynamic per-32 MXFP4 activation quantization (round to nearest E2M1, UE8M0 scale)."""
    shape = x.shape
    g = x.float().reshape(-1, 32)
    amax = g.abs().amax(dim=-1, keepdim=True).clamp_min(2.0 ** -126)
    scale = torch.pow(2.0, torch.floor(torch.log2(amax)) - 2.0)
    grid = torch.tensor(FP4_GRID, dtype=torch.float32, device=x.device)
    q = (g / scale).clamp(-6.0, 6.0)
    idx = (q.abs().unsqueeze(-1) - grid).abs().argmin(dim=-1)
    return (grid[idx] * q.sign() * scale).reshape(shape)
'''


def def_moe_mxfp4():
    name = f"moe_mxfp4_routed_topk9_e{N_EXPERTS}_h{HIDDEN}_i{MOE_I}"
    ref = '''import torch
''' + REF_MXFP4 + f'''

@torch.no_grad()
def run(hidden_states, topk_ids, topk_weights, w13, w13_scale, w2, w2_scale):
    """Routed MoE, precomputed routing (aiter fused_moe interface). 9 = 8 routed + 1 fused
    shared expert (id 256). Activations are MXFP4-quantized before each GEMM (a4w4).
    w13 rows [0, I) = gate, [I, 2I) = up; out = sum_k w_k * W2_e(silu(gate) * up).
    routed_scaling_factor (2.5) is NOT applied here (sglang applies it outside the kernel)."""
    E, I, H = {N_EXPERTS}, {MOE_I}, {HIDDEN}
    M, K = topk_ids.shape
    x = quant_dequant_mxfp4(hidden_states)
    out = torch.zeros(M, H, dtype=torch.float32, device=hidden_states.device)
    ids = topk_ids.to(torch.int64)
    for e in torch.unique(ids).tolist():
        rows, slots = torch.nonzero(ids == e, as_tuple=True)
        W13 = dequant_mxfp4(w13[e], w13_scale[e])  # [2I, H]
        W2 = dequant_mxfp4(w2[e], w2_scale[e])     # [H, I]
        g1 = x[rows] @ W13.t()
        act = torch.nn.functional.silu(g1[:, :I]) * g1[:, I:]
        o = quant_dequant_mxfp4(act) @ W2.t()
        out.index_add_(0, rows, o * topk_weights[rows, slots].unsqueeze(1))
    return out.to(torch.bfloat16)
'''
    return name, "moe", {
        "name": name, "op_type": "moe",
        "description": "GLM-5.2-MXFP4 decode-leg routed MoE per rank (TP4, EP1): MXFP4 (E2M1 + UE8M0, "
                       "group 32) weights and dynamically MXFP4-quantized activations (a4w4), SwiGLU, "
                       "257 experts = 256 routed + 1 fused shared (id 256, weight 1.0), top-9 per token. "
                       "Production kernel: aiter fused_moe_ -> gemm1_a4w4 / gemm2_a4w4 (+ mxfp4_moe_sort). "
                       "Input rows = all 4 attention-DP ranks' tokens after the DP all-gather "
                       "(seq_len = 4 x 6 x bs_graph for target verify). Weights here are in logical "
                       "layout; aiter consumes a pre-shuffled layout (weight preparation is offline).",
        "tags": ["stage:decode", "phase:target_verify", "quantization:float4_e2m1fn_mxfp4", "fused"] + MODEL_TAGS,
        "axes": {
            "seq_len": axis_var("Tokens entering the MoE on this rank (gathered over attention-DP)."),
            "num_experts": axis_const(N_EXPERTS, "256 routed + 1 fused shared expert."),
            "top_k": axis_const(TOPK + 1, "8 routed + 1 shared."),
            "hidden_size": axis_const(HIDDEN, "Model hidden size."),
            "hidden_size_packed": axis_const(HIDDEN // 2, "hidden_size / 2 (two E2M1 per byte)."),
            "hidden_size_scales": axis_const(HIDDEN // 32, "hidden_size / 32 (one UE8M0 per 32)."),
            "gemm1_out_size": axis_const(2 * MOE_I, "2 x intermediate (gate + up)."),
            "intermediate_size": axis_const(MOE_I, "moe_intermediate_size 2048 / TP4."),
            "intermediate_size_packed": axis_const(MOE_I // 2, "intermediate / 2."),
            "intermediate_size_scales": axis_const(MOE_I // 32, "intermediate / 32."),
        },
        "inputs": {
            "hidden_states": T(["seq_len", "hidden_size"], "bfloat16", "Gathered MoE input."),
            "topk_ids": T(["seq_len", "top_k"], "int32", "Expert ids; column 8 is always 256 (shared)."),
            "topk_weights": T(["seq_len", "top_k"], "float32", "Combine weights; column 8 is 1.0."),
            "w13": T(["num_experts", "gemm1_out_size", "hidden_size_packed"], "int8", "Packed E2M1 pairs (read as uint8)."),
            "w13_scale": T(["num_experts", "gemm1_out_size", "hidden_size_scales"], "int8", "UE8M0 exponents (read as uint8)."),
            "w2": T(["num_experts", "hidden_size", "intermediate_size_packed"], "int8", "Packed E2M1 pairs (read as uint8)."),
            "w2_scale": T(["num_experts", "hidden_size", "intermediate_size_scales"], "int8", "UE8M0 exponents (read as uint8)."),
        },
        "outputs": {"output": T(["seq_len", "hidden_size"], "bfloat16", "sum_k w_k * expert_k(x).")},
        "reference": ref,
    }


def def_moe_bf16():
    name = f"moe_bf16_routed_topk9_e{N_EXPERTS}_h{HIDDEN}_i{MOE_I}"
    ref = f'''import torch


@torch.no_grad()
def run(hidden_states, topk_ids, topk_weights, w13, w2):
    """Routed bf16 MoE, precomputed routing. w13 rows [0, I) = gate, [I, 2I) = up."""
    I, H = {MOE_I}, {HIDDEN}
    M, K = topk_ids.shape
    x = hidden_states.float()
    out = torch.zeros(M, H, dtype=torch.float32, device=x.device)
    ids = topk_ids.to(torch.int64)
    for e in torch.unique(ids).tolist():
        rows, slots = torch.nonzero(ids == e, as_tuple=True)
        g1 = x[rows] @ w13[e].float().t()
        act = torch.nn.functional.silu(g1[:, :I]) * g1[:, I:]
        o = act @ w2[e].float().t()
        out.index_add_(0, rows, o * topk_weights[rows, slots].unsqueeze(1))
    return out.to(torch.bfloat16)
'''
    return name, "moe", {
        "name": name, "op_type": "moe",
        "description": "GLM-5.2 MTP draft layer (GlmMoeDsaForCausalLMNextN) routed MoE per rank: weights are "
                       "bf16 in this checkpoint (not MXFP4); production kernel aiter fused_moe_ -> CK "
                       "ck_moe_stage1/2. 257 experts (256 + fused shared), top-9. Used by draft decode "
                       "(seq_len = 4 x bs_graph, 4 steps / iteration) and draft extend (seq_len = 4 x 6 x bs_graph).",
        "tags": ["stage:decode", "phase:draft_decode", "phase:draft_extend", "fused"] + MODEL_TAGS,
        "axes": {
            "seq_len": axis_var("Tokens entering the MoE on this rank (gathered over attention-DP)."),
            "num_experts": axis_const(N_EXPERTS, "256 routed + 1 fused shared expert."),
            "top_k": axis_const(TOPK + 1, "8 routed + 1 shared."),
            "hidden_size": axis_const(HIDDEN, "Model hidden size."),
            "gemm1_out_size": axis_const(2 * MOE_I, "2 x intermediate (gate + up)."),
            "intermediate_size": axis_const(MOE_I, "2048 / TP4."),
        },
        "inputs": {
            "hidden_states": T(["seq_len", "hidden_size"], "bfloat16", "Gathered MoE input."),
            "topk_ids": T(["seq_len", "top_k"], "int32", "Expert ids; column 8 is always 256 (shared)."),
            "topk_weights": T(["seq_len", "top_k"], "float32", "Combine weights; column 8 is 1.0."),
            "w13": T(["num_experts", "gemm1_out_size", "hidden_size"], "bfloat16", "Gate+up weights."),
            "w2": T(["num_experts", "hidden_size", "intermediate_size"], "bfloat16", "Down weights."),
        },
        "outputs": {"output": T(["seq_len", "hidden_size"], "bfloat16", "sum_k w_k * expert_k(x).")},
        "reference": ref,
    }


def def_routing():
    name = f"moe_routing_sigmoid_bias_topk{TOPK}_e{N_ROUTED}_shared1"
    ref = f'''import torch


@torch.no_grad()
def run(router_logits, correction_bias):
    """noaux_tc routing (n_group = topk_group = 1): select top-8 by sigmoid(logit) + bias,
    weight = sigmoid(logit) renormalized over the 8; append fused shared expert 256 with weight 1.0.
    Production: aiter biased_grouped_topk + triton _fused_append_shared_experts_kernel."""
    s = torch.sigmoid(router_logits.float())
    _, ids = torch.topk(s + correction_bias.float(), {TOPK}, dim=-1)
    w = torch.gather(s, 1, ids)
    w = w / w.sum(dim=-1, keepdim=True)
    M = router_logits.shape[0]
    shared_id = torch.full((M, 1), {N_ROUTED}, dtype=torch.int64, device=ids.device)
    shared_w = torch.ones((M, 1), dtype=torch.float32, device=ids.device)
    return torch.cat([ids, shared_id], 1).to(torch.int32), torch.cat([w, shared_w], 1)
'''
    return name, "moe", {
        "name": name, "op_type": "moe",
        "description": "GLM-5.2 MoE routing on the decode leg: biased sigmoid top-8 over 256 routed experts "
                       "(scoring_func sigmoid, noaux_tc, renormalize) + fused shared-expert append. "
                       "Two small kernels in production; candidate for fusion with the router GEMM.",
        "tags": ["stage:decode", "phase:target_verify", "fused"] + MODEL_TAGS,
        "axes": {"seq_len": axis_var("Gathered tokens on this rank."),
                 "num_routed": axis_const(N_ROUTED, "Routed experts."),
                 "top_k_out": axis_const(TOPK + 1, "8 routed + 1 shared.")},
        "inputs": {"router_logits": T(["seq_len", "num_routed"], "bfloat16", "Router GEMM output."),
                   "correction_bias": T(["num_routed"], "bfloat16", "e_score_correction_bias.")},
        "outputs": {"topk_ids": T(["seq_len", "top_k_out"], "int32", "Expert ids."),
                    "topk_weights": T(["seq_len", "top_k_out"], "float32", "Combine weights.")},
        "reference": ref,
    }


def def_sparse_mla():
    name = f"dsa_sparse_mla_fp8_h{HEADS}_ckv{KV_LORA}_kpe{ROPE}_topk{TOPK_ATTN}_ps{PAGE}"
    ref = f'''import torch


@torch.no_grad()
def run(q_nope, q_rope, kv_cache, sparse_indices, sm_scale):
    """Sparse MLA over an fp8 latent KV cache (k_scale = v_scale = 1.0 in the deployment).
    kv_cache[p, o, :512] = latent ckv (also V), [512:576] = rope key. sparse_indices are global
    token slots (page * 64 + offset), -1 = padding."""
    T, H, D = q_nope.shape
    kv = kv_cache.reshape(-1, kv_cache.shape[-1]).float()
    invalid = sparse_indices < 0
    idx = sparse_indices.clamp_min(0).long()
    K = kv[idx]                                   # [T, topk, 576]
    q = torch.cat([q_nope.float(), q_rope.float()], dim=-1)   # [T, H, 576]
    logits = torch.einsum("thd,tkd->thk", q, K) * sm_scale
    logits = logits.masked_fill(invalid.unsqueeze(1), float("-inf"))
    p = torch.softmax(logits, dim=-1)
    out = torch.einsum("thk,tkd->thd", p, K[..., :{KV_LORA}])
    return out.to(torch.bfloat16)
'''
    return name, "dsa_paged", {
        "name": name, "op_type": "dsa_paged",
        "description": "GLM-5.2 DSA sparse MLA attention per rank (64 heads, attn_tp 1) as run on the decode leg "
                       "for TARGET_VERIFY and DRAFT_EXTEND_V2 (speculative_attention_mode=prefill -> "
                       "dsa_prefill_backend=triton: _sparse_mla_split_k_kernel + _sparse_mla_reduce_kernel, "
                       "KV_SPLITS 2, fp8 dot). q is fp8 (written by fused_qk_rope_cat_and_cache_mla). "
                       "sm_scale = (192 + 64) ** -0.5; the kernel folds log2(e) in (qk_scale 0.0902). "
                       "num_tokens = 6 x bs_graph.",
        "tags": ["stage:decode", "phase:target_verify", "phase:draft_extend", "sparse:topk",
                 "quantization:float8_e4m3fn"] + MODEL_TAGS,
        "axes": {"num_tokens": axis_var("Query tokens on this rank (6 per request)."),
                 "num_qo_heads": axis_const(HEADS, "Heads per rank."),
                 "head_dim_ckv": axis_const(KV_LORA, "Latent (V) dim."),
                 "head_dim_kpe": axis_const(ROPE, "Rope dim."),
                 "head_dim_kv": axis_const(KV_LORA + ROPE, "Stored KV row (ckv + kpe)."),
                 "page_size": axis_const(PAGE, "Tokens per page."),
                 "topk": axis_const(TOPK_ATTN, "Selected KV tokens per query."),
                 "num_pages": axis_var("Pages in the per-rank KV pool.")},
        "inputs": {"q_nope": T(["num_tokens", "num_qo_heads", "head_dim_ckv"], "float8_e4m3fn", "Absorbed q (latent)."),
                   "q_rope": T(["num_tokens", "num_qo_heads", "head_dim_kpe"], "float8_e4m3fn", "Rope part of q."),
                   "kv_cache": T(["num_pages", "page_size", "head_dim_kv"], "float8_e4m3fn", "Latent KV cache."),
                   "sparse_indices": T(["num_tokens", "topk"], "int32", "Global token slots from the indexer."),
                   "sm_scale": T(None, "float32", "Softmax scale 0.0625.")},
        "outputs": {"output": T(["num_tokens", "num_qo_heads", "head_dim_ckv"], "bfloat16", "Latent output (before W_UV).")},
        "constraints": ["sparse_indices.shape[-1] == topk"],
        "reference": ref,
    }


def def_indexer():
    name = f"dsa_topk_indexer_fp8_h{IDX_HEADS}_d{IDX_DIM}_topk{TOPK_ATTN}_ps{PAGE}"
    ref = f'''import torch


def dequant_fp8_kv_cache(k_index_cache_fp8):
    """deep_gemm/aiter paged index-K layout: per page, page_size*128 fp8 bytes then page_size fp32 scales."""
    k = k_index_cache_fp8.view(torch.uint8)
    num_pages, page_size, _, head_dim_sf = k.shape
    head_dim = head_dim_sf - 4
    flat = k.view(num_pages, page_size * head_dim_sf)
    vals = flat[:, : page_size * head_dim].contiguous().view(num_pages, page_size, head_dim).view(torch.float8_e4m3fn).float()
    scale = flat[:, page_size * head_dim:].contiguous().view(num_pages, page_size, 4).view(torch.float32)
    return vals * scale


@torch.no_grad()
def run(q_index_fp8, k_index_cache_fp8, weights, seq_lens, block_table):
    """Same semantics as upstream dsa_topk_indexer_fp8_h64_d128_topk2048_ps64, with 32 index heads.
    One row per query token: in TARGET_VERIFY / DRAFT_EXTEND_V2 a request contributes 6 rows with
    causal lengths L+1 .. L+6 (sglang seqlens_expanded)."""
    rows, page_size, topk = q_index_fp8.shape[0], {PAGE}, {TOPK_ATTN}
    q = q_index_fp8.float()
    K_all = dequant_fp8_kv_cache(k_index_cache_fp8)
    out = torch.full((rows, topk), -1, dtype=torch.int32, device=q.device)
    for b in range(rows):
        L = int(seq_lens[b].item())
        if L == 0:
            continue
        pages = block_table[b, : (L + page_size - 1) // page_size].long()
        K = K_all[pages].reshape(-1, K_all.shape[-1])[:L]
        score = (torch.relu(q[b] @ K.t()) * weights[b][:, None]).sum(0)
        k = min(topk, L)
        _, idx = torch.topk(score, k)
        out[b, :k] = (pages[idx // page_size] * page_size + idx % page_size).to(torch.int32)
    return (out,)
'''
    return name, "dsa_paged", {
        "name": name, "op_type": "dsa_paged",
        "description": "GLM-5.2 DSA indexer per rank (32 index heads, d 128, fp8 paged index-K cache) as run "
                       "on the decode leg for TARGET_VERIFY (21 of 78 layers: 0,1,2,6,10,...,74) and "
                       "DRAFT_EXTEND_V2. Production = _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle "
                       "(logits f32[rows, max_seq_len]) + sgl_kernel fast_topk_transform_fused. The only "
                       "decode op whose cost grows with context length.",
        "tags": ["stage:indexer", "stage:decode", "phase:target_verify", "phase:draft_extend",
                 "sparse:topk", "quantization:float8_e4m3fn"] + MODEL_TAGS,
        "axes": {"batch_size": axis_var("Query rows (6 per request)."),
                 "num_index_heads": axis_const(IDX_HEADS, "index_n_heads."),
                 "index_head_dim": axis_const(IDX_DIM, "index_head_dim."),
                 "page_size": axis_const(PAGE, "Tokens per page."),
                 "topk": axis_const(TOPK_ATTN, "index_topk."),
                 "max_num_pages": axis_var("Block-table width."),
                 "num_pages": axis_var("Pages in the per-rank index-K pool."),
                 "kv_cache_num_heads": axis_const(1, "MQA."),
                 "head_dim_with_scale": axis_const(IDX_DIM + 4, "128 fp8 + 4-byte fp32 scale.")},
        "inputs": {"q_index_fp8": T(["batch_size", "num_index_heads", "index_head_dim"], "float8_e4m3fn", "Index queries."),
                   "k_index_cache_fp8": T(["num_pages", "page_size", "kv_cache_num_heads", "head_dim_with_scale"], "int8", "Paged index-K cache (read as uint8)."),
                   "weights": T(["batch_size", "num_index_heads"], "float32", "Per-head weights."),
                   "seq_lens": T(["batch_size"], "int32", "Causal length per row."),
                   "block_table": T(["batch_size", "max_num_pages"], "int32", "Page table per row.")},
        "outputs": {"topk_indices": T(["batch_size", "topk"], "int32", "Global token slots, -1 padding.")},
        "constraints": ["topk <= max_num_pages * page_size"],
        "reference": ref,
    }


def def_gemm(n, k, role):
    name = f"gemm_bf16_n{n}_k{k}"
    return name, "gemm", {
        "name": name, "op_type": "gemm",
        "description": f"bf16 GEMM C = A @ B.T, GLM-5.2 decode leg per rank: {role}.",
        "tags": ["stage:decode"] + MODEL_TAGS,
        "axes": {"M": axis_var("Rows (tokens)."), "N": axis_const(n, "Output features."),
                 "K": axis_const(k, "Input features.")},
        "inputs": {"A": T(["M", "K"], "bfloat16", "Activations."), "B": T(["N", "K"], "bfloat16", "Weight.")},
        "outputs": {"C": T(["M", "N"], "bfloat16", "A @ B.T.")},
        "reference": "import torch\n\n\ndef run(A, B):\n    return torch.matmul(A.float(), B.float().T).to(torch.bfloat16)\n",
    }


def def_absorb_bmm(k, n, role):
    name = f"mla_absorb_bmm_fp8w_h{HEADS}_k{k}_n{n}"
    ref = '''import torch


@torch.no_grad()
def run(A, B, B_scale):
    """Per-head GEMM out[t, h] = A[t, h] @ (B[h] * B_scale). Production kernel quantizes A to fp8
    per token group on the fly (_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant)."""
    return torch.einsum("thk,hkn->thn", A.float(), B.float() * B_scale).to(torch.bfloat16)
'''
    return name, "gemm", {
        "name": name, "op_type": "gemm",
        "description": f"GLM-5.2 MLA weight-absorption batched GEMM per rank ({role}); fp8 per-tensor-scaled "
                       "weights, bf16 activations, 64 heads. Called twice per layer (W_UK and W_UV).",
        "tags": ["stage:decode", "quantization:float8_e4m3fn"] + MODEL_TAGS,
        "axes": {"num_tokens": axis_var("Tokens."), "num_heads": axis_const(HEADS, "Heads per rank."),
                 "K": axis_const(k, "Reduction dim."), "N": axis_const(n, "Output dim.")},
        "inputs": {"A": T(["num_tokens", "num_heads", "K"], "bfloat16", "Activations (strided view in production)."),
                   "B": T(["num_heads", "K", "N"], "float8_e4m3fn", "Per-head weight."),
                   "B_scale": T(None, "float32", "Per-tensor weight scale.")},
        "outputs": {"C": T(["num_tokens", "num_heads", "N"], "bfloat16", "Per-head product.")},
        "reference": ref,
    }


def def_add_rmsnorm():
    name = f"fused_add_rmsnorm_h{HIDDEN}"
    ref = f'''import torch


@torch.no_grad()
def run(hidden_states, residual, weight):
    x = hidden_states.float() + residual.float()
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-5) * weight.float()
    return y.to(hidden_states.dtype)
'''
    return name, "rmsnorm", {
        "name": name, "op_type": "rmsnorm",
        "description": "Fused add + RMSNorm, hidden 6144, eps 1e-5 (GLM-5.2). Production aiter add_rmsnorm; "
                       "sglang also writes the updated residual (x) back in place.",
        "tags": ["stage:decode", "fused"] + MODEL_TAGS,
        "axes": {"batch_size": axis_var("Tokens."), "hidden_size": axis_const(HIDDEN, "Hidden size.")},
        "inputs": {"hidden_states": T(["batch_size", "hidden_size"], "bfloat16", "Input."),
                   "residual": T(["batch_size", "hidden_size"], "bfloat16", "Residual."),
                   "weight": T(["hidden_size"], "bfloat16", "Gamma.")},
        "outputs": {"output": T(["batch_size", "hidden_size"], "bfloat16", "Normalized output.")},
        "reference": ref,
    }


# ---- workloads -------------------------------------------------------------------------------
def uid(*parts):
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "glm52-decode-workset/" + "/".join(map(str, parts))))


class Writer:
    def __init__(self, out):
        self.out = Path(out)
        self.meta = {}
        self.lines = {}
        self.seen = {}

    def definition(self, name, op_type, d):
        p = self.out / "definitions" / op_type / f"{name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(d, indent=2) + "\n")

    def blob(self, op_type, name, tag, tensors):
        rel = f"blob/workloads/{op_type}/{name}/{name}_{tag}.safetensors"
        write_safetensors(self.out / rel, tensors)
        return rel

    def workload(self, name, op_type, axes, inputs, meta, key):
        """One line per unique (definition, axes, inputs); every place it is used is a 'use'."""
        sig = json.dumps([name, axes, inputs], sort_keys=True)
        if sig in self.seen:
            self.meta[self.seen[sig]]["uses"].append(meta)
            return
        u = uid(name, key)
        self.seen[sig] = u
        self.lines.setdefault((op_type, name), []).append(
            {"definition": name, "workload": {"uuid": u, "axes": axes, "inputs": inputs},
             "solution": None, "evaluation": None})
        self.meta[u] = {"definition": name, "axes": axes, "uses": [meta]}

    def flush(self):
        for (op_type, name), lines in self.lines.items():
            p = self.out / "workloads" / op_type / f"{name}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("".join(json.dumps(l) + "\n" for l in lines))


def st(rel, key):
    return {"type": "safetensors", "path": rel, "tensor_key": key}


R = {"type": "random"}


def routing_blob(rng, m, shared=True):
    ids, ws = [], []
    for _ in range(m):
        chosen = rng.sample(range(N_ROUTED), TOPK)
        s = [1 / (1 + math.exp(-rng.gauss(0, 1))) for _ in chosen]
        tot = sum(s)
        ids += chosen + [N_ROUTED]
        ws += [x / tot for x in s] + [1.0]
    return {"topk_ids": ("I32", [m, TOPK + 1], i32(ids)), "topk_weights": ("F32", [m, TOPK + 1], f32(ws))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--mu", type=float, default=9.0)
    ap.add_argument("--sigma", type=float, default=5 / 3)
    ap.add_argument("--lo", type=int, default=4)
    ap.add_argument("--hi", type=int, default=14)
    ap.add_argument("--min-prob", type=float, default=0.005, help="drop bs_graph values rarer than this")
    ap.add_argument("--ctx-samples", type=int, default=2, help="context draws per bs for context-dependent ops")
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    per_rank, graph = batch_distribution(a.mu, a.sigma, a.lo, a.hi, 200_000, rng)
    bs_list = [b for b, p in graph.items() if p >= a.min_prob]
    w = Writer(a.out)

    defs = {}
    for make in (def_moe_mxfp4, def_moe_bf16, def_routing, def_sparse_mla, def_indexer, def_add_rmsnorm):
        name, op, d = make()
        defs[make.__name__] = (name, op)
        w.definition(name, op, d)
    gemms = {"o_proj": (HIDDEN, 16384, "attention o_proj (64 heads x 256 v_head -> 6144)"),
             "q_b": (16384, 2048, "q_b_proj (q_lora 2048 -> 64 x 256)"),
             "qkv_a": (2624, HIDDEN, "fused_qkv_a_proj_with_mqa (6144 -> q_lora 2048 + kv_lora 512 + rope 64)"),
             "router": (N_ROUTED, HIDDEN, "MoE router gate (gathered tokens)"),
             "lm_head": (VOCAB_SHARD, HIDDEN, "draft lm_head, vocab-parallel shard (gathered tokens)")}
    for key, (n, k, role) in gemms.items():
        name, op, d = def_gemm(n, k, role)
        defs[key] = (name, op)
        w.definition(name, op, d)
    for key, (k, n, role) in {"w_uk": (192, KV_LORA, "W_UK: q_nope 192 -> latent 512"),
                              "w_uv": (KV_LORA, 256, "W_UV: latent 512 -> v_head 256")}.items():
        name, op, d = def_absorb_bmm(k, n, role)
        defs[key] = (name, op)
        w.definition(name, op, d)

    # The MXFP4 MoE scales are the one large constant blob (~76 MB, shared by every workload).
    moe_name, moe_op = defs["def_moe_mxfp4"]
    srng = random.Random(a.seed + 1)
    table = bytes(118 + (i % 9) for i in range(256))   # UE8M0 exponents 2^-9 .. 2^-1
    scale_rel = w.blob(moe_op, moe_name, "weight_scales", {
        "w13_scale": ("I8", [N_EXPERTS, 2 * MOE_I, HIDDEN // 32], srng.randbytes(N_EXPERTS * 2 * MOE_I * HIDDEN // 32).translate(table)),
        "w2_scale": ("I8", [N_EXPERTS, HIDDEN, MOE_I // 32], srng.randbytes(N_EXPERTS * HIDDEN * MOE_I // 32).translate(table)),
    })

    for bs in bs_list:
        p = graph[bs]
        Tv, Gv = DRAFT * bs, DP * DRAFT * bs        # verify / draft-extend tokens; MoE rows
        Td, Gd = bs, DP * bs                         # one draft-decode step
        base = {"bs_graph": bs, "probability": p}
        def add(key, axes, inputs, phase, calls, tag=""):
            name, op = defs[key]
            w.workload(name, op, axes, inputs, {**base, "phase": phase, "calls_per_iteration": calls},
                       f"{bs}/{phase}/{tag}")
        # target verify (78 layers; 75 MoE; 3 dense MLP layers not exported)
        rel = w.blob(moe_op, moe_name, f"bs{bs}_verify", routing_blob(rng, Gv))
        add("def_moe_mxfp4", {"seq_len": Gv},
            {"hidden_states": R, "topk_ids": st(rel, "topk_ids"), "topk_weights": st(rel, "topk_weights"),
             "w13": R, "w13_scale": st(scale_rel, "w13_scale"), "w2": R, "w2_scale": st(scale_rel, "w2_scale")},
            "target_verify", 75)
        add("def_routing", {"seq_len": Gv}, {"router_logits": R, "correction_bias": R}, "target_verify", 75)
        add("router", {"M": Gv}, {"A": R, "B": R}, "target_verify", 75)
        for g in ("o_proj", "q_b", "qkv_a"):
            add(g, {"M": Tv}, {"A": R, "B": R}, "target_verify", 78)
            add(g, {"M": Tv}, {"A": R, "B": R}, "draft_extend", 1)
            add(g, {"M": Td}, {"A": R, "B": R}, "draft_decode", 4)
        for g in ("w_uk", "w_uv"):
            add(g, {"num_tokens": Tv}, {"A": R, "B": R, "B_scale": {"type": "scalar", "value": 1.0}}, "target_verify", 78)
        add("def_add_rmsnorm", {"batch_size": Tv}, {"hidden_states": R, "residual": R, "weight": R}, "target_verify", 156)
        # draft MTP layer: bf16 MoE + lm_head
        bf_name, bf_op = defs["def_moe_bf16"]
        rel_e = w.blob(bf_op, bf_name, f"bs{bs}_extend", routing_blob(rng, Gv))
        rel_d = w.blob(bf_op, bf_name, f"bs{bs}_decode", routing_blob(rng, Gd))
        for rel_x, m, phase, calls in ((rel_e, Gv, "draft_extend", 1), (rel_d, Gd, "draft_decode", 4)):
            add("def_moe_bf16", {"seq_len": m},
                {"hidden_states": R, "topk_ids": st(rel_x, "topk_ids"), "topk_weights": st(rel_x, "topk_weights"),
                 "w13": R, "w2": R}, phase, calls)
        add("lm_head", {"M": Gv}, {"A": R, "B": R}, "draft_extend", 1)
        add("lm_head", {"M": Gd}, {"A": R, "B": R}, "draft_decode", 4)
        # context-dependent: sparse MLA indices and the indexer
        for c in range(a.ctx_samples):
            ctx = sample_contexts(bs, rng)
            pages = list(range(POOL_PAGES))
            rng.shuffle(pages)
            seq_rows, tables, idx_rows, cursor = [], [], [], 0
            need = [(L + DRAFT + PAGE - 1) // PAGE for L in ctx]
            width = max(need)
            for L, npg in zip(ctx, need):
                own = pages[cursor:cursor + npg]
                cursor += npg
                for j in range(DRAFT):
                    Lj = L + j + 1
                    seq_rows.append(Lj)
                    tables.append(own + [0] * (width - npg))
                    sel = rng.sample(range(Lj), TOPK_ATTN)
                    idx_rows += [own[s // PAGE] * PAGE + s % PAGE for s in sel]
            tag = f"bs{bs}_ctx{c}"
            name_i, op_i = defs["def_indexer"]
            rel_i = w.blob(op_i, name_i, tag, {
                "seq_lens": ("I32", [Tv], i32(seq_rows)),
                "block_table": ("I32", [Tv, width], i32([x for row in tables for x in row]))})
            name_s, op_s = defs["def_sparse_mla"]
            rel_s = w.blob(op_s, name_s, tag, {"sparse_indices": ("I32", [Tv, TOPK_ATTN], i32(idx_rows))})
            cmeta = {"context_per_request": ctx}
            for phase, calls in (("target_verify", 21), ("draft_extend", 1)):
                name, op = defs["def_indexer"]
                w.workload(name, op, {"batch_size": Tv, "max_num_pages": width, "num_pages": POOL_PAGES},
                           {"q_index_fp8": R, "k_index_cache_fp8": R, "weights": R,
                            "seq_lens": st(rel_i, "seq_lens"), "block_table": st(rel_i, "block_table")},
                           {**base, "phase": phase, "calls_per_iteration": calls,
                            "probability_within_bs": 1 / a.ctx_samples, **cmeta}, f"{bs}/{phase}/{tag}")
            for phase, calls in (("target_verify", 78), ("draft_extend", 1)):
                name, op = defs["def_sparse_mla"]
                w.workload(name, op, {"num_tokens": Tv, "num_pages": POOL_PAGES},
                           {"q_nope": R, "q_rope": R, "kv_cache": R, "sparse_indices": st(rel_s, "sparse_indices"),
                            "sm_scale": {"type": "scalar", "value": SM_SCALE}},
                           {**base, "phase": phase, "calls_per_iteration": calls,
                            "probability_within_bs": 1 / a.ctx_samples, **cmeta}, f"{bs}/{phase}/{tag}")
    w.flush()
    (Path(a.out) / "glm52_workload_meta.json").write_text(json.dumps({
        "batch_model": {"per_rank_running_requests": f"round(Normal({a.mu}, {a.sigma:.4f})) clamped to [{a.lo}, {a.hi}]",
                        "dp_ranks": DP, "bs_graph": "snap_up(max over ranks), captured list " + str(CAPTURED_BS[:14]),
                        "per_rank_B_probability": per_rank, "bs_graph_probability": graph,
                        "exported_bs_graph": bs_list, "seed": a.seed},
        "context_model": {"lognormal_median": CTX_MEDIAN, "lognormal_sigma": CTX_SIGMA,
                          "per_rank_total_cap": CTX_RANK_CAP, "clamp": [4096, 1_000_000]},
        "workloads": w.meta}, indent=1) + "\n")
    print(json.dumps({"bs_graph": graph, "exported": bs_list, "workloads": len(w.meta)}))


if __name__ == "__main__":
    main()
