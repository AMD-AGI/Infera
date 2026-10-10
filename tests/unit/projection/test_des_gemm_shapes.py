###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A replayed trace reports the GEMM shapes it runs, as the kernels see them.

Two things are pinned here. The shapes are the engine's (weights sharded and
fused the way vLLM / SGLang launch them on one TP rank), not the cost model's
FLOP-equivalent sizing. And ``M`` comes from the steps the DES actually
executed, so every token the replay pushes through a layer is counted once.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from types import SimpleNamespace

from infera.projection.core.projection.inference_projection import des as des_mod
from infera.projection.core.projection.inference_projection import gemm_shapes as gs
from infera.projection.core.projection.training_config import ModelConfig, ModelParallelConfig


def _gqa(**kw) -> ModelConfig:
    base = dict(
        num_layers=4,
        hidden_size=4096,
        padded_vocab_size=128000,
        ffn_hidden_size=14336,
        num_attention_heads=32,
        kv_channels=128,
        group_query_attention=True,
        num_query_groups=8,
        swiglu=True,
    )
    base.update(kw)
    return ModelConfig(**base)


def _moe(**kw) -> ModelConfig:
    base = dict(
        num_experts=128,
        moe_router_topk=4,
        moe_ffn_hidden_size=2880,
        moe_pattern=[1, 1, 1, 1],
        moe_shared_expert_intermediate_size=0,
    )
    base.update(kw)
    return _gqa(**base)


def _cfg(mc, tp=8, ep=1, dp=1, linear=None, expert=None):
    mp = ModelParallelConfig(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        attention_data_parallel_size=dp,
    )
    req = SimpleNamespace(linear_weight_dtype=linear, moe_expert_dtype=expert)
    return SimpleNamespace(model_config=mc, model_parallel_config=mp, request_config=req)


def _by_op(cfg, tokens, rows=None):
    return {g.op: (g, calls) for g, calls in gs.step_gemms(cfg, tokens, rows or tokens)}


def test_attention_and_mlp_shard_n_not_m():
    """TP slices heads and the FFN; every rank still carries all the tokens."""
    ops = _by_op(_cfg(_gqa(), tp=8), tokens=300, rows=12)
    qkv, n_attn = ops["attn_qkv"]
    assert (qkv.m, qkv.n, qkv.k, n_attn) == (300, (4 + 2 * 1) * 128, 4096, 4)
    assert ops["attn_o"][0].k == 4 * 128
    assert (ops["mlp_gate_up"][0].m, ops["mlp_gate_up"][0].n) == (300, 2 * 14336 // 8)
    assert (ops["mlp_down"][0].n, ops["mlp_down"][0].k) == (4096, 14336 // 8)
    head, n_head = ops["lm_head"]
    assert (head.m, head.n, n_head) == (12, 128000 // 8, 1)  # only sampled rows


def test_kv_heads_are_replicated_when_fewer_than_tp():
    ops = _by_op(_cfg(_gqa(num_query_groups=4), tp=8), tokens=16)
    assert ops["attn_qkv"][0].n == (4 + 2 * 1) * 128


def test_routed_experts_are_one_grouped_gemm_over_the_local_experts():
    cfg = _cfg(_moe(moe_shared_expert_intermediate_size=2048), tp=8, ep=8, expert="mxfp4")
    ops = _by_op(cfg, tokens=64)
    assert "mlp_gate_up" not in ops  # every layer is MoE
    gate_up, n_moe = ops["moe_gate_up"]
    assert (gate_up.m, gate_up.n, gate_up.k) == (64, 2 * 2880, 4096)  # EP=TP: experts unsharded
    assert (gate_up.groups, gate_up.topk, gate_up.rows_per_expert, n_moe) == (16, 4, 2, 4)
    assert gate_up.dtype == "mxfp4"
    assert ops["moe_router"][0].n == 128 and ops["moe_router"][0].dtype == "bf16"
    assert ops["shared_gate_up"][0].n == 2 * 2048 // 8
    assert ops["shared_gate_up"][0].dtype == "bf16"  # the linears keep their own precision


def test_experts_tensor_shard_when_ep_is_narrower_than_tp():
    gate_up = _by_op(_cfg(_moe(), tp=8, ep=2), tokens=64)["moe_gate_up"][0]
    assert (gate_up.n, gate_up.groups) == (2 * 2880 // 4, 64)


def test_mla_is_the_fused_down_projection_then_per_head_up_projections():
    mc = _gqa(
        multi_latent_attention=True,
        num_attention_heads=128,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        hidden_size=7168,
    )
    ops = _by_op(_cfg(mc, tp=8), tokens=32)
    assert (ops["attn_qkv_a"][0].n, ops["attn_qkv_a"][0].k) == (1536 + 512 + 64, 7168)
    assert (ops["attn_q_b"][0].n, ops["attn_q_b"][0].k) == (16 * 192, 1536)
    assert (ops["attn_kv_b"][0].n, ops["attn_kv_b"][0].k) == (16 * 256, 512)
    assert (ops["attn_o"][0].n, ops["attn_o"][0].k) == (7168, 16 * 128)


def test_attention_dp_splits_attention_tokens_but_not_the_mlp():
    ops = _by_op(_cfg(_gqa(), tp=8, dp=4), tokens=400)
    qkv = ops["attn_qkv"][0]
    assert (qkv.m, qkv.n) == (100, (16 + 2 * 4) * 128)  # 1/4 of the tokens over tp/dp = 2
    assert ops["mlp_gate_up"][0].m == 400


def test_linear_attention_layers_launch_no_attention_projections():
    ops = _by_op(_cfg(_gqa(linear_attention_layers=3), tp=8), tokens=8)
    assert ops["attn_qkv"][1] == 1
    assert ops["mlp_gate_up"][1] == 4


def test_declared_linear_precision_wins():
    ops = _by_op(_cfg(_gqa(), linear="fp8"), tokens=8)
    assert ops["attn_qkv"][0].dtype == "fp8"
    assert ops["lm_head"][0].dtype == "bf16"


# ── the replay ───────────────────────────────────────────────────────────────


@dataclass
class _Req:
    input_seq_len: int = 1024
    output_seq_len: int = 16
    max_concurrency: int = 4
    max_num_batched_tokens: int = 300
    chunked_prefill_size: int = 256
    speculative_num_tokens: int = 0
    speculative_acceptance_rate: float = 0.0
    tokenize_overhead_us: float = 0.0
    detokenize_overhead_us: float = 0.0
    linear_weight_dtype: str | None = None
    moe_expert_dtype: str | None = None

    def resolved_max_concurrency(self) -> int:
        return self.max_concurrency

    def resolved_max_context_len(self) -> int:
        return self.input_seq_len + self.output_seq_len

    def resolved_prefix_cache_hit_rate(self) -> float:
        return 0.0


class _Projector:
    def __init__(self, cfg):
        self.cfg = cfg

    def decode_step_latency_ms(self, batch, ctx, q_len):
        return 2.0 + 0.05 * batch

    def mixed_step_latency_ms(self, num_decode, prefill_tokens, ctx, prefill_kv, q_len):
        return 2.0 + 0.05 * num_decode + 0.002 * prefill_tokens


def _replay(trace: gs.GemmTrace, clients=4, requests=12):
    req = _Req(max_concurrency=clients)
    cfg = SimpleNamespace(
        request_config=req,
        model_config=_gqa(),
        model_parallel_config=ModelParallelConfig(tensor_model_parallel_size=8),
    )
    with gs.recording(trace):
        des_mod.simulate_once(
            cfg,
            _Projector(cfg),
            rate_per_s=0.0,
            arrival_model="closed",
            num_requests=requests,
            warmup_frac=0.0,
            closed_loop_clients=clients,
        )
    return req


def test_every_token_the_replay_runs_is_counted_once():
    trace = gs.GemmTrace()
    req = _replay(trace)
    rows = trace.shapes()
    qkv_tokens = sum(r["m"] * r["calls"] for r in rows if r["op"] == "attn_qkv") // 4
    # Each prompt is prefilled once, then every output token after the first decodes.
    assert qkv_tokens == 12 * (req.input_seq_len + req.output_seq_len - 1)
    assert trace.num_steps > 0
    # Chunked prefill mixes with decodes, so M is not one value.
    assert len({r["m"] for r in rows if r["op"] == "attn_qkv"}) > 3
    assert all(r["est_us"] is None for r in rows)  # no simulator, no estimate


def test_a_discarded_pass_and_a_paused_block_leave_no_steps():
    trace = gs.GemmTrace()
    with gs.recording(trace):
        with gs.pass_capture():
            _record(5)
        with gs.pass_capture() as kept:
            _record(7)
        kept.commit()
        with gs.paused():
            _record(9)
    assert {r["m"] for r in trace.shapes() if r["op"] == "attn_qkv"} == {7}


def _record(tokens: int) -> None:
    trace = gs.active()
    if trace is not None:
        trace.record(SimpleNamespace(cfg=_cfg(_gqa())), "engine", tokens, 1)


def test_the_output_files(tmp_path):
    trace = gs.GemmTrace()
    _replay(trace)
    out = trace.write(str(tmp_path / "g.csv"))
    with open(tmp_path / "g.csv") as fh:
        assert len(list(csv.DictReader(fh))) == len(out)
    trace.write(str(tmp_path / "g.json"))
    doc = json.loads((tmp_path / "g.json").read_text())
    assert doc["steps"] == trace.num_steps
    qkv = next(r for r in doc["by_weight"] if r["op"] == "attn_qkv")
    assert qkv["m_p50"] <= qkv["m_p90"] <= qkv["m_p99"] <= qkv["m_max"]
    assert qkv["distinct_m"] == len([r for r in doc["shapes"] if r["op"] == "attn_qkv"])
