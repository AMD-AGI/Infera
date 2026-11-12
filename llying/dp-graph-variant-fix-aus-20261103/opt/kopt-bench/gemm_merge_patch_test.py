#!/usr/bin/env python3
"""Mechanics of kopt_gemm_merge (kit/overlay-gemmmerge) without an engine: the import hook patches SGLang's classes,
and on a minimal attention + indexer built from SGLang's own ReplicatedLinear (GLM-5.2 shapes) the merged weights are
views of one tensor, the merged GEMM's slices equal the separate projections, the indexer projections take the stashed
columns only for the same input, and nothing changes outside capture or below the row threshold.

Usage (1 GPU, production image, SGLANG_USE_AITER=1, KOPT_GEMM_MERGE=1 [KOPT_GEMM_MERGE_QB=1]):
    python3 gemm_merge_patch_test.py
"""
import os
import sys
import types

sys.path.insert(0, "/kopt_overlay_extra")
import kopt_gemm_merge as kgm  # noqa: E402

kgm.install()

import torch  # noqa: E402

import sglang.srt.models.deepseek_v2 as dsv2  # noqa: E402
from sglang.srt.layers.linear import ReplicatedLinear  # noqa: E402
import sglang.srt.model_executor.runner as runner  # noqa: E402

H, Q_LORA, KV_A, HEADS, QK_HEAD, IDX_HEADS, IDX_DIM = 6144, 2048, 576, 64, 256, 32, 128
ok = True


def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"{'PASS' if cond else 'FAIL'}  {name}", flush=True)


def lin(n_in, n_out):
    m = ReplicatedLinear(n_in, n_out, bias=False, params_dtype=torch.bfloat16).cuda()
    with torch.no_grad():
        m.weight.normal_(0, 0.02)
    return m


def mode(verify):
    return types.SimpleNamespace(forward_mode=types.SimpleNamespace(is_decode=lambda: False,
                                                                     is_target_verify=lambda: verify))


check("prepare_qkv_latent patched", dsv2.DeepseekV2AttentionMLA.prepare_qkv_latent.__module__ == "kopt_gemm_merge")
check("q_b_proj_forward patched", dsv2.DeepseekV2AttentionMLA.q_b_proj_forward.__module__ == "kopt_gemm_merge")
check("post_load_weights patched", dsv2.DeepseekV2ForCausalLM.post_load_weights.__module__ == "kopt_gemm_merge")

torch.manual_seed(0)
attn = types.SimpleNamespace(
    fused_qkv_a_proj_with_mqa=lin(H, Q_LORA + KV_A), q_b_proj=lin(Q_LORA, HEADS * QK_HEAD), skip_topk=False,
    is_nextn=False, num_local_heads=HEADS, qk_head_dim=QK_HEAD, q_lora_rank=Q_LORA,
    indexer=types.SimpleNamespace(wk=lin(H, IDX_DIM), weights_proj=lin(H, IDX_HEADS), wq_b=lin(Q_LORA, IDX_HEADS * IDX_DIM)),
)
attn._use_min_latency_fused_a_gemm = False
attn.has_fused_proj = True
attn._use_min_latency_q_b_gemm = False
attn._q_b_proj_verified_shape = False
idx = attn.indexer
ref_w = {k: m.weight.detach().clone() for k, m in
         [("fq", attn.fused_qkv_a_proj_with_mqa), ("wk", idx.wk), ("wp", idx.weights_proj), ("qb", attn.q_b_proj),
          ("wq", idx.wq_b)]}
model = types.SimpleNamespace(model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=attn)]))
kgm._build(model)
m = attn._kopt_qkva
check("weights are row views of the merged tensor",
      all(w.data_ptr() == m.holder.weight.data_ptr() + off * H * 2 for w, off in
          [(attn.fused_qkv_a_proj_with_mqa.weight, 0), (idx.wk.weight, Q_LORA + KV_A),
           (idx.weights_proj.weight, Q_LORA + KV_A + IDX_DIM)]))
check("weight values unchanged", torch.equal(attn.fused_qkv_a_proj_with_mqa.weight, ref_w["fq"])
      and torch.equal(idx.wk.weight, ref_w["wk"]) and torch.equal(idx.weights_proj.weight, ref_w["wp"]))

prep = lambda x, fb: dsv2.DeepseekV2AttentionMLA.prepare_qkv_latent(attn, x, fb)  # noqa: E731
qbf = lambda q: dsv2.DeepseekV2AttentionMLA.q_b_proj_forward(attn, q)  # noqa: E731
runner.get_is_capture_mode = lambda: True
for rows in (48, 64, 88, 96):
    x = torch.randn(rows, H, device="cuda", dtype=torch.bfloat16)
    sep_q = attn.fused_qkv_a_proj_with_mqa.quant_method.apply(attn.fused_qkv_a_proj_with_mqa, x, None)
    sep_k = idx.wk.quant_method.apply(idx.wk, x, None)
    sep_w = idx.weights_proj.quant_method.apply(idx.weights_proj, x, None)
    refq = x.float() @ ref_w["fq"].float().t()
    qkv = prep(x, mode(True))
    armed = attn._kopt_armed
    check(f"rows {rows}: armed = {rows >= kgm._MIN_ROWS}", armed == (rows >= kgm._MIN_ROWS))
    k = idx.wk(x)[0]
    w = idx.weights_proj(x)[0]
    err = lambda a, b: ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()  # noqa: E731
    check(f"rows {rows}: qkv_a rel err vs fp32 {err(qkv, refq):.1e} (separate {err(sep_q, refq):.1e})",
          err(qkv, refq) < max(4 * err(sep_q, refq), 1e-2))
    check(f"rows {rows}: wk / weights_proj match the separate GEMMs (rel {err(k, sep_k):.1e} / {err(w, sep_w):.1e})",
          err(k, sep_k) < 2e-2 and err(w, sep_w) < 2e-2)
    if armed:
        check(f"rows {rows}: wk output is a column slice of the merged GEMM", k.stride(0) == sum(m.sizes))
        check(f"rows {rows}: stash consumed", not idx._kopt_stash)
        prep(x, mode(True))
        y = x.clone()
        k2 = idx.wk(y)[0]
        check(f"rows {rows}: another input falls back to a separate GEMM", k2.stride(0) == IDX_DIM and torch.equal(k2, sep_k))
    q_lora = torch.randn(rows, Q_LORA, device="cuda", dtype=torch.bfloat16)
    prep(x, mode(True))
    q = qbf(q_lora)
    wq = idx.wq_b(q_lora)[0]
    sep_qb = attn.q_b_proj.quant_method.apply(attn.q_b_proj, q_lora, None).view(-1, HEADS, QK_HEAD)
    sep_wq = idx.wq_b.quant_method.apply(idx.wq_b, q_lora, None)
    check(f"rows {rows}: q_b_proj / wq_b match the separate GEMMs (rel {err(q, sep_qb):.1e} / {err(wq, sep_wq):.1e}), "
          f"merged {'on' if getattr(attn, '_kopt_qb', None) is not None and armed else 'off'}",
          err(q, sep_qb) < 2e-2 and err(wq, sep_wq) < 2e-2)
runner.get_is_capture_mode = lambda: False
x = torch.randn(96, H, device="cuda", dtype=torch.bfloat16)
prep(x, mode(True))
check("outside capture: not armed", not attn._kopt_armed and not idx._kopt_stash)
print("ALL PASS" if ok else "SOME FAILED", flush=True)
