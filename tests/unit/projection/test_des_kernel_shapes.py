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
from infera.projection.core.projection.training_config import ModelParallelConfig

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
