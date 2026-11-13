#!/usr/bin/env python3
"""Bytes a decode rank has to stream from HBM per verify step (GLM-5.2-MXFP4, decode TP over N GPUs with DP attention)
and the time floor at a given bandwidth. Shapes from the checkpoint config.json; precisions from its quantization
`exclude` list (attention / indexer / dense-MLP / lm_head linears stay BF16, only routed + shared experts are MXFP4).

Usage: hw_model.py [TP=8]
"""
import json, sys
tp = int(sys.argv[1]) if len(sys.argv) > 1 else 8
c = json.load(open("/perf_apps/xiaobo/models/GLM-5.2-MXFP4/config.json"))
H = c["hidden_size"]; L = c["num_hidden_layers"]; nd = c["first_k_dense_replace"]; LM = L - nd
E = c["n_routed_experts"] + c["n_shared_experts"]; I = c["moe_intermediate_size"]; topk = c["num_experts_per_tok"]
nh = c["num_attention_heads"]; qr = c["q_lora_rank"]; kvr = c["kv_lora_rank"]
dq = c["qk_nope_head_dim"] + c["qk_rope_head_dim"]; dv = c["v_head_dim"]; rope = c["qk_rope_head_dim"]
n_full = sum(1 for t in c["indexer_types"] if t == "full")
ih, idim = c["index_n_heads"], c["index_head_dim"]
MB = 1e6
mxfp4 = lambda n: n * 0.5 * (1 + 1 / 16)   # 4-bit elements + one E8M0 scale byte per 32
bf16 = lambda n: n * 2
ops = {
    "fused_qkv_a (q_a + kv_a) BF16": (L, bf16((qr + kvr + rope) * H)),
    "q_b_proj BF16": (L, bf16(nh * dq * qr)),
    "o_proj BF16": (L, bf16(H * nh * dv)),
    "absorb W_kc + W_vc (kv_b) FP8": (L, nh * (c["qk_nope_head_dim"] * kvr + kvr * dv)),
    "indexer wq_b + wk + weights_proj BF16": (n_full, bf16(ih * idim * qr + idim * H + ih * H)),
    "dense MLP (layers 0-2) BF16, TP split": (nd, bf16(3 * c["intermediate_size"] * H) / tp),
    "lm_head BF16, TP split": (1, bf16(c["vocab_size"] * H) / tp),
    "MoE router gate BF16": (LM, bf16(c["n_routed_experts"] * H)),
    "MoE experts gate_up MXFP4, TP split": (LM, mxfp4(E * 2 * I * H) / tp),
    "MoE experts down MXFP4, TP split": (LM, mxfp4(E * H * I) / tp),
}
tot = 0.0
print(f"TP{tp}: per verify step, per rank")
for k, (n, b) in ops.items():
    print(f"  {k:42s} {b / MB:8.1f} MB x {n:2d} = {n * b / 1e9:6.2f} GB")
    tot += n * b
attn = sum(n * b for k, (n, b) in ops.items() if "MoE experts" not in k)
moe = tot - attn
print(f"  total {tot / 1e9:.1f} GB = MoE experts {moe / 1e9:.1f} + everything else {attn / 1e9:.1f}")
print(f"  per MoE layer: experts {moe / LM / MB:.1f} MB; attention projections (qkv_a + q_b + o_proj) {sum(ops[k][1] for k in list(ops)[:3]) / MB:.1f} MB")
for bw in (4.0, 5.0, 6.0, 8.0):
    print(f"  at {bw:.0f} TB/s: all experts hit {tot / bw / 1e9:6.2f} ms; MoE {moe / bw / 1e9:5.2f} ms + rest {attn / bw / 1e9:5.2f} ms")
print("\nexpected share of the routed experts hit by M tokens (uniform routing, top-%d of %d; an upper bound when routing is skewed):" % (topk, c["n_routed_experts"]))
for M in (4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 640, 768):
    cov = 1 - (1 - topk / c["n_routed_experts"]) ** M
    print(f"  M={M:4d}: {100 * cov:5.1f}% -> {(cov * c['n_routed_experts'] + 1) / E * moe / LM / MB:6.1f} MB per layer")
kv = (kvr + rope)
print(f"\nKV gather per attention row: {c['index_topk']} entries x {kv} B = {c['index_topk'] * kv / MB:.2f} MB per layer, {L * c['index_topk'] * kv / MB:.0f} MB per step")
print(f"index keys per cached context token: {idim} B per full-indexer layer, {n_full} layers -> {n_full * idim / 1e3:.2f} KB per token per step (+ the draft_extend pass)")
