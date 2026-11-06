###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A replayed trace reports every kernel it runs, as one rank launches it.

Beyond the GEMMs, attention is the shape a tuner needs most and the one a
token count cannot give: it depends on how many sequences a step carries and
how far back each one attends. Those come from the steps the DES executed, so
the tests pin both the per-rank geometry and that the replay's sequences reach
the report.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from infera.projection.core.projection.inference_projection import gemm_shapes as gs
from infera.projection.core.projection.inference_projection import kernel_shapes as ks
from infera.projection.core.projection.training_config import ModelConfig, ModelParallelConfig

from .test_des_gemm_shapes import _gqa, _moe, _replay


def _cfg(mc, tp=8, ep=1, dp=1, dcp=1, pp=1, kv_dtype="bf16", window=None):
    mp = ModelParallelConfig(
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
        attention_data_parallel_size=dp,
        decode_context_parallel_size=dcp,
        pipeline_model_parallel_size=pp,
    )
    req = SimpleNamespace(kv_cache_dtype=kv_dtype, linear_weight_dtype=None, moe_expert_dtype=None)
    return SimpleNamespace(model_config=mc, model_parallel_config=mp, request_config=req)


def _kernels(cfg, decode=(), prefill=(), q_len=1, sampled=None):
    tokens = len(decode) * q_len + sum(q for q, _ in prefill)
    rows = sampled or (len(decode) * q_len + len(prefill))
    step = ks.make_step(cfg, tokens, rows, list(decode), list(prefill), q_len)
    out: dict[str, list] = {}
    for k, calls in ks.step_kernels(cfg, step):
        out.setdefault(k.op, []).append((k, dict(k.dims), calls))
    return out


def test_decode_attention_is_the_rank_s_heads_over_each_sequence_s_span():
    ops = _kernels(_cfg(_gqa(), tp=8, kv_dtype="fp8"), decode=[1000, 3000, 200])
    [(k, d, calls)] = ops["attn_decode"]
    assert k.kind == "attention" and k.dtype == "fp8" and calls == 4
    assert (d["batch"], d["q"], d["heads"], d["kv_heads"], d["qk_dim"]) == (3, 1, 4, 1, 128)
    # 3001 is the longest span attended (this step's token included), bucketed up.
    assert (d["max_kv"], d["mean_kv"]) == (3072, 1536)
    assert "attn_prefill" not in ops


def test_a_mixed_step_launches_prefill_and_decode_attention_and_one_gemm_batch():
    ops = _kernels(_cfg(_gqa()), decode=[500] * 6, prefill=[(200, 0), (100, 4000)])
    [(_, pre, _)] = ops["attn_prefill"]
    assert (pre["batch"], pre["q"], pre["max_q"], pre["max_kv"]) == (2, 320, 256, 4352)
    assert ops["attn_decode"][0][1]["batch"] == 6
    assert ops["attn_qkv"][0][1]["m"] == 306  # the linears see every token at once


def test_mla_prefills_per_head_and_decodes_over_the_absorbed_latent():
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
    ops = _kernels(_cfg(mc, tp=8), decode=[100], prefill=[(64, 0)])
    pre, dec = ops["attn_prefill"][0][1], ops["attn_decode"][0][1]
    assert (pre["heads"], pre["kv_heads"], pre["qk_dim"], pre["v_dim"]) == (16, 16, 192, 128)
    assert (dec["heads"], dec["kv_heads"], dec["qk_dim"], dec["v_dim"]) == (16, 1, 576, 512)
    assert ops["kv_cache_write"][0][1]["dim"] == 576
    assert {"q_a_rmsnorm", "kv_a_rmsnorm"} <= set(ops)


def test_windowed_layers_are_their_own_rows_and_decode_reads_the_window_if_the_pool_keeps_it():
    mc = _gqa(sink_sliding_window=128, sink_window_even_layers_only=True)
    ops = _kernels(_cfg(mc), decode=[5000], prefill=[(512, 4096)])
    pre = {d["window"]: (d["max_kv"], calls) for _, d, calls in ops["attn_prefill"]}
    assert pre == {0: (4608, 2), 128: (4608, 2)}  # half the 4 layers each; the kernel masks
    dec = {d["window"]: d["max_kv"] for _, d, _ in ops["attn_decode"]}
    assert dec == {0: 5120, 128: 5120}
    mc = _gqa(
        sink_sliding_window=128, sink_window_even_layers_only=True, sink_window_decode_capped=True
    )
    dec = {d["window"]: d["max_kv"] for _, d, _ in _kernels(_cfg(mc), decode=[5000])["attn_decode"]}
    assert dec == {0: 5120, 128: 256}


def test_attention_dp_splits_sequences_and_dcp_splits_their_spans():
    ops = _kernels(_cfg(_gqa(), tp=8, dp=4), decode=[1000] * 10)
    d = ops["attn_decode"][0][1]
    assert (d["batch"], d["heads"], d["kv_heads"]) == (3, 16, 4)  # tp/dp = 2 ranks of heads
    assert {k.op for k, _, _ in ops["tp_allgather"]} == {"tp_allgather"}
    assert "tp_allreduce" not in ops
    d = _kernels(_cfg(_gqa(), tp=8, dcp=4), decode=[4000])["attn_decode"][0][1]
    assert (d["max_kv"], d["heads"]) == (1024, 16)


def test_moe_routing_activation_and_dispatch_are_listed_per_moe_layer():
    ops = _kernels(_cfg(_moe(), tp=8, ep=8), decode=[100] * 64)
    assert ops["moe_topk"][0][1] == {"tokens": 64, "experts": 128, "topk": 4}
    act = ops["moe_silu_and_mul"][0]
    assert (act[1]["rows"], act[1]["width"], act[2]) == (32, 2880, 4)  # 64*4 rows over 8 ranks
    dispatch = ops["ep_dispatch"][0][1]
    assert dispatch == {"ranks": 8, "bytes": 64 * 4096 * 4 * 2}
    assert ops["tp_allreduce"][0][2] == 2 * 4
    assert "silu_and_mul" not in ops  # no dense layer


def test_gated_delta_net_layers_list_their_own_kernels():
    mc = _gqa(
        linear_attention_layers=3,
        linear_attention_kind="gdn",
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_attention_conv_kernel=4,
    )
    ops = _kernels(_cfg(mc, tp=4), decode=[100] * 2, prefill=[(64, 0)])
    assert ops["gdn_prefill"][0][1]["key_heads"] == 4 and ops["gdn_prefill"][0][2] == 3
    assert ops["gdn_decode"][0][1]["value_heads"] == 8
    assert ops["causal_conv1d"][0][1] == {
        "tokens": 66,
        "channels": 2 * 4 * 128 + 8 * 128,
        "kernel": 4,
    }
    assert ops["gdn_in_proj_qkvz"][0][1]["n"] == 2 * 4 * 128 + 2 * 8 * 128
    assert ops["attn_decode"][0][2] == 1  # one full-attention layer left


def test_the_replay_s_decodes_all_reach_decode_attention():
    trace = ks.KernelTrace()
    req = _replay(trace)
    rows = trace.kernels()
    decodes = sum(r["batch"] * r["calls"] for r in rows if r["op"] == "attn_decode") // 4
    assert decodes == 12 * (req.output_seq_len - 1)
    # Spans run 1025..1040 (prompt plus the tokens so far): one 256-token bucket.
    assert {r["max_kv"] for r in rows if r["op"] == "attn_decode"} == {1280}
    gemm = gs.GemmTrace()
    _replay(gemm)
    assert sorted((r["op"], r["m"], r["calls"]) for r in gemm.shapes()) == sorted(
        (r["op"], r["m"], r["calls"]) for r in rows if r["kind"] == "gemm"
    )


def test_a_rerun_pass_keeps_the_sequences():
    trace = ks.KernelTrace()
    cfg = _cfg(_gqa())
    with gs.recording(trace):
        with gs.pass_capture() as kept:
            gs.active().record(SimpleNamespace(cfg=cfg), "engine", 2, 2, decode=[10, 20])
        kept.commit()
    assert any(r["op"] == "attn_decode" for r in trace.kernels())


def test_the_output_files(tmp_path):
    trace = ks.KernelTrace()
    _replay(trace)
    out = trace.write_kernels(str(tmp_path / "k.csv"))
    with open(tmp_path / "k.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == len(out)
    assert {"kind", "op", "batch", "max_kv", "m", "bytes", "calls"} <= set(rows[0])
    trace.write_kernels(str(tmp_path / "k.json"))
    doc = json.loads((tmp_path / "k.json").read_text())
    assert doc["steps"] == trace.num_steps
    decode = next(r for r in doc["by_op"] if r["op"] == "attn_decode")
    assert decode["distinct_shapes"] == len([r for r in doc["kernels"] if r["op"] == "attn_decode"])
    assert {r["kind"] for r in doc["kernels"]} >= {"gemm", "attention", "norm", "collective"}


# ── the layers the first version left out ────────────────────────────────────


def _v4(**kw):
    base = dict(
        num_layers=4,
        hidden_size=7168,
        padded_vocab_size=129280,
        ffn_hidden_size=18432,
        num_attention_heads=128,
        num_query_groups=1,
        group_query_attention=True,
        kv_channels=512,
        qk_pos_emb_head_dim=64,
        q_lora_rank=1536,
        o_lora_rank=1024,
        o_groups=16,
        hc_mult=4,
        hybrid_attention_enabled=True,
        compress_ratios=[128, 4, 128, 0],
        index_topk=1024,
        index_n_heads=64,
        index_head_dim=128,
        attn_sliding_window=128,
        num_experts=384,
        moe_router_topk=6,
        moe_ffn_hidden_size=3072,
        moe_pattern=[1, 1, 1, 1],
        swiglu=True,
    )
    base.update(kw)
    return ModelConfig(**base)


def test_deepseek_v4_lists_its_own_projections_compressors_and_mixers():
    ops = _kernels(_cfg(_v4(), tp=8), decode=[8000] * 4)
    shape = {op: rows[0][1] for op, rows in ops.items() if rows[0][0].kind == "gemm"}
    assert (shape["attn_q_a"]["n"], shape["attn_q_b"]["n"], shape["attn_kv"]["n"]) == (
        1536,
        16 * 512,
        512,
    )
    o_a = shape["attn_o_a"]
    assert (o_a["groups"], o_a["n"], o_a["k"]) == (
        2,
        1024,
        128 * 512 // 16,
    )  # 16 groups over 8 ranks
    assert shape["attn_o_b"]["k"] == 2 * 1024
    assert "attn_qkv" not in ops
    compressors = {d["n"]: calls for _, d, calls in ops["compressor_kv_gate"]}
    assert compressors == {4 * 512: 1, 2 * 512: 2}  # one CSA, two HCA layers
    assert ops["idx_q_b"][0][2] == 1 and ops["hc_mixer"][0][2] == 2 * 4
    assert ops["hc_head"][0][2] == 1 and ops["hc_sinkhorn"][0][2] == 2 * 4


def test_deepseek_v4_attends_over_its_window_and_compressed_pool():
    ops = _kernels(_cfg(_v4(), tp=8), decode=[8000] * 4)
    by_ratio = {
        d["compress"]: (d["max_kv"], d["pool_kv"], calls) for _, d, calls in ops["attn_decode"]
    }
    # 8001 bucketed to 8192: HCA sees all 8192/128 entries, CSA the indexer's top 1024 of 2048.
    assert by_ratio == {0: (256, 0, 1), 4: (256, 1024, 1), 128: (256, 64, 2)}
    dec = ops["attn_decode"][0][1]
    assert (dec["heads"], dec["kv_heads"], dec["qk_dim"]) == (16, 1, 512)
    idx = ops["indexer_decode"][0]
    assert (idx[1]["q"], idx[1]["max_kv"], idx[1]["heads"], idx[2]) == (1, 2048, 64, 1)
    assert ops["indexer_topk"][0][1] == {"rows": 4, "candidates": 2048, "k": 1024}
    assert {d["tokens"] for _, d, _ in ops["pool_cache_write"]} == {1}


def _sparse_mla(**kw):
    base = dict(
        multi_latent_attention=True,
        num_attention_heads=64,
        qk_head_dim=128,
        qk_pos_emb_head_dim=64,
        v_head_dim=128,
        q_lora_rank=2048,
        kv_lora_rank=512,
        hidden_size=6144,
        sparse_index_n_heads=32,
        sparse_index_head_dim=128,
        sparse_index_layers=2,
        sparse_index_dtype="fp8",
        index_topk=2048,
    )
    base.update(kw)
    return _gqa(**base)


def test_a_top_k_sparse_model_indexes_its_indexed_layers_and_attends_over_its_top_k():
    ops = _kernels(_cfg(_sparse_mla(), tp=8), decode=[10000] * 2)
    assert ops["indexer_decode"][0][2] == 2  # 2 of the 4 layers carry an indexer
    idx = ops["indexer_decode"][0][1]
    assert (idx["heads"], idx["dim"], idx["max_kv"]) == (32, 128, 10240)
    assert ops["indexer_decode"][0][0].dtype == "fp8"
    assert all(d["topk"] == 2048 for _, d, _ in ops["attn_decode"])
    assert ops["idx_q"][0][1]["k"] == 2048 and ops["idx_q"][0][2] == 2  # from the Q latent
    assert ops["index_cache_write"][0][1]["dim"] == 128


def test_kda_layers_list_their_own_projections():
    mc = _gqa(
        num_layers=4,
        linear_attention_layers=3,
        linear_attention_kind="kda",
        linear_attention_head_dim=128,
        linear_attention_full_rank_gate=True,
    )
    ops = _kernels(_cfg(mc, tp=8), decode=[100])
    pk = (32 // 8) * 128
    assert ops["kda_q"][0][1]["n"] == pk and ops["kda_q"][0][2] == 3
    assert (ops["kda_decay_a"][0][1]["n"], ops["kda_decay_b"][0][1]["k"]) == (128, 128)
    assert ops["kda_gate"][0][1]["n"] == pk and "kda_gate_a" not in ops
    assert ops["kda_out_proj"][0][1]["k"] == pk
    assert ops["gated_rmsnorm"][0][2] == 3


def test_an_output_gate_widens_the_qkv_projection_or_stands_beside_mla():
    ops = _kernels(_cfg(_gqa(attention_output_gate=True), tp=8), decode=[100])
    assert ops["attn_qkv"][0][1]["n"] == (4 + 2 + 4) * 128
    assert ops["attn_gate_mul"][0][1]["width"] == 4 * 128
    per_head = _kernels(
        _cfg(_gqa(attention_output_gate=True, attention_output_gate_per_head=True)), decode=[1]
    )
    assert per_head["attn_qkv"][0][1]["n"] == (4 + 2) * 128 + 4
    mla = _kernels(_cfg(_sparse_mla(attention_output_gate=True), tp=8), decode=[100])
    assert mla["attn_gate"][0][1]["n"] == 8 * 128


def test_a_latent_moe_runs_its_experts_on_the_latent():
    ops = _kernels(_cfg(_moe(moe_latent_hidden_size=1024), tp=8, ep=8), decode=[100] * 8)
    assert ops["moe_gate_up"][0][1]["k"] == 1024 and ops["moe_down"][0][1]["n"] == 1024
    assert ops["moe_latent_down"][0][1]["n"] == 1024 and ops["moe_latent_up"][0][1]["k"] == 1024


def test_quantized_activations_are_cast_in_front_of_each_low_precision_gemm():
    cfg = _cfg(_moe(), tp=8, ep=8)
    cfg.request_config = SimpleNamespace(
        kv_cache_dtype="fp8",
        linear_weight_dtype="fp8",
        moe_expert_dtype="fp8",
        resolved_act_quant_dtype=lambda fp8: "fp8",
    )
    ops = _kernels(cfg, decode=[100] * 64)
    assert ops["attn_qkv_act_quant"][0][1] == {"rows": 64, "width": 4096}
    assert ops["moe_down_act_quant"][0][1] == {"rows": 32, "width": 2880}  # the rank's expert rows
    assert "moe_router_act_quant" not in ops and "lm_head_act_quant" not in ops


def test_each_pipeline_stage_counts_its_own_layers():
    cfg = _cfg(_moe(num_layers=4, moe_pattern=[0, 1, 1, 1]), tp=8, pp=2)
    step = ks.make_step(cfg, 16, 16, [100] * 16)
    stages = [
        {k.op: calls for k, calls in ks.step_kernels(cfg, step, c)} for c in ks.lp.stage_counts(cfg)
    ]
    assert stages[0]["mlp_gate_up"] == 1 and stages[0]["moe_gate_up"] == 1
    assert stages[1]["moe_gate_up"] == 2 and "mlp_gate_up" not in stages[1]
    assert "embedding" in stages[0] and "embedding" not in stages[1]
    assert "lm_head" in stages[1] and "sampling" in stages[1] and "lm_head" not in stages[0]
    assert stages[0]["pp_send_recv"] == 1 and "pp_send_recv" not in stages[1]
    assert stages[0]["rmsnorm"] == 4 and stages[1]["rmsnorm"] == 5  # the final norm too


def test_layer_types_land_where_the_model_puts_them():
    qwen = _cfg(_gqa(num_layers=8, linear_attention_freq=4, linear_attention_kind="gdn"))
    assert [x.attn for x in ks.lp.plan(qwen)] == ["linear"] * 3 + ["full"] + ["linear"] * 3 + [
        "full"
    ]
    oss = _cfg(_gqa(sink_sliding_window=128, sink_window_even_layers_only=True))
    assert [x.attn for x in ks.lp.plan(oss)] == ["window", "full", "window", "full"]
    v4 = _cfg(_v4())
    assert [x.attn for x in ks.lp.plan(v4)] == ["v4_hca", "v4_csa", "v4_hca", "v4_swa"]
    assert [x.indexed for x in ks.lp.plan(v4)] == [False, True, False, False]


class _SpecProjector:
    def __init__(self, cfg, drafter, k=3, rates=(0.8, 0.6, 0.4)):
        self.cfg, self._d, self._k, self._rates = cfg, drafter, k, list(rates)

    def _drafter(self):
        return self._d

    def _spec_k(self):
        return self._k

    def _spec_rates(self, k=None):
        return self._rates


def test_a_speculating_replay_reports_its_drafter_s_passes():
    from infera.projection.core.projection.inference_projection.speculative import (
        acceptance_length,
        drafter_for,
    )

    cfg = _cfg(_moe(mtp_num_layers=1), tp=8, ep=8)
    proj = _SpecProjector(cfg, drafter_for("mtp", target_vocab=128000, mtp_layers=1))
    trace = ks.KernelTrace()
    with gs.recording(trace):
        gs.active().record(proj, "engine", 8 * 4, 8 * 4, decode=[500] * 8, q_len=4)
    rows = [r for r in trace.kernels() if r["pool"] == "engine/draft"]
    first_q = round(acceptance_length([0.8, 0.6, 0.4]))
    qkv = {r["m"]: r["calls"] for r in rows if r["op"] == "attn_qkv"}
    assert qkv == {8 * first_q: 1, 8: 2}  # one extend pass, then k - 1 single-token passes
    assert {r["m"] for r in rows if r["op"] == "draft_eh_proj"} == {8 * first_q, 8}
    assert any(r["op"] == "moe_gate_up" for r in rows)  # the MTP layer carries the experts
    target = [r for r in trace.kernels() if r["pool"] == "engine" and r["op"] == "attn_decode"]
    assert target[0]["q"] == 4  # the verify step itself


class _Backend:
    hbm_bandwidth_gbps = 8000.0
    peak_tflops_bf16 = 2500.0

    def is_available(self):
        return True

    def simulate_gemm(self, m, n, k, dtype, batch=1):
        return SimpleNamespace(forward_time_ms=1e-3)

    def simulate_sdpa(self, **kw):
        self.sdpa_args = kw
        return SimpleNamespace(forward_time_ms=2e-3)


def test_every_kernel_carries_a_time_and_its_source():
    backend = _Backend()
    cfg = _cfg(_moe(), tp=8, ep=8)
    proj = SimpleNamespace(cfg=cfg, _gemm=backend, _sdpa=backend, _comm=None)
    trace = ks.KernelTrace()
    with gs.recording(trace):
        gs.active().record(proj, "engine", 70, 7, decode=[3000] * 6, prefill=[(64, 0)])
    rows = trace.kernels()
    assert all(r["est_us"] is not None for r in rows if r["kind"] != "collective"), [
        r["op"] for r in rows if r["est_us"] is None
    ]
    src = {r["kind"]: r["est_source"] for r in rows}
    assert (src["gemm"], src["attention"], src["norm"], src["moe"]) == (
        "origami",
        "sdpa",
        "roofline",
        "roofline",
    )
    assert backend.sdpa_args["seq_len_kv"] >= 3000
    rms = next(r for r in rows if r["op"] == "rmsnorm")
    # 3 passes of 70 x 4096 bf16 over 56.6% of 8 TB/s.
    assert abs(rms["est_us"] - 1e6 * 3 * 70 * 4096 * 2 / (8000e9 * 0.566)) < 1e-3
