###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""dLLM-Sim: the properties a refinement-step model must have to be trusted.

There is no MI355X measurement of a diffusion LM to assert milliseconds
against yet, so these assert the structure the cost has to follow: what a
bidirectional step costs relative to a causal one, how the cache modes order,
what quantizing a collective buys on each link, the limits of the overlap
model, where memory runs out, and that the two frontends count the same
arithmetic. The numbers they pin are formulas, not fits.
"""

from __future__ import annotations

import json

import pytest

from infera.projection.core.projection.inference_projection.dllm import (
    CommConfig,
    ContentionConfig,
    DLLMCommModel,
    DLLMHardware,
    DLLMModelSpec,
    DLLMStepModel,
    DLLMWorkload,
    InferaSimBackend,
    Parallel,
    SeqShape,
    build_stage_graph,
    dllm_presets,
    load_model,
    max_batch,
    memory_report,
    simulate_serving,
    simulate_static,
)
from infera.projection.core.projection.inference_projection.dllm.contention import overlap
from infera.projection.core.projection.inference_projection.dllm.hardware import dtype_bytes
from infera.projection.core.projection.inference_projection.dllm.ops import ATTENTION, GEMM

HW = DLLMHardware.from_arch("mi355x")


def _tiny() -> DLLMModelSpec:
    return DLLMModelSpec(
        name="tiny",
        num_layers=2,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=8,
        num_query_groups=2,
        vocab_size=1000,
    )


def _flops(graph, kind):
    return sum(o.flops * o.count for o in graph.ops if o.kind == kind)


# -- frontends ----------------------------------------------------------------


def test_presets_load_with_their_dimensions():
    assert {"llada_8B", "dream_7B"} <= set(dllm_presets())
    llada = load_model("llada_8B")
    assert (llada.num_layers, llada.hidden_size, llada.vocab_size) == (32, 4096, 126464)
    assert llada.num_query_groups == 32
    dream = load_model("dream_7B")
    assert (dream.num_query_groups, dream.vocab_size) == (4, 152064)


def test_megatron_preset_loads_as_a_dllm_model():
    spec = load_model("llama3.1_8B")
    assert spec.hidden_size == 4096 and spec.num_layers == 32


def test_bidirectional_attention_is_the_full_product():
    spec = _tiny()
    g = build_stage_graph(spec, [SeqShape(64, 64, 64)], Parallel(), first=True, last=True)
    per_layer = 4.0 * 64 * 64 * spec.num_attention_heads * spec.head_dim
    assert _flops(g, ATTENTION) == pytest.approx(per_layer * spec.num_layers)


def test_head_runs_over_every_predicted_position():
    spec = _tiny()
    full = build_stage_graph(spec, [SeqShape(64, 64, 64)], Parallel(), first=True, last=True)
    part = build_stage_graph(spec, [SeqShape(64, 64, 8)], Parallel(), first=True, last=True)
    head = {o.name: o for o in full.ops}["lm_head"].flops
    assert head == pytest.approx(2.0 * 64 * spec.vocab_size * spec.hidden_size)
    assert {o.name: o for o in part.ops}["lm_head"].flops == pytest.approx(head / 8)


def test_tensor_parallel_shards_gemm_flops():
    spec = _tiny()
    seqs = [SeqShape(64, 64, 64)]
    one = build_stage_graph(spec, seqs, Parallel(tp=1), first=True, last=True)
    two = build_stage_graph(spec, seqs, Parallel(tp=2), first=True, last=True)
    assert _flops(two, GEMM) == pytest.approx(_flops(one, GEMM) / 2)
    assert [c.kind for c in two.comm] == ["tp_allreduce", "tp_allreduce"]


def test_low_precision_weights_pay_an_activation_cast():
    spec = _tiny()
    spec.weight_dtype = "fp8"
    g = build_stage_graph(spec, [SeqShape(64, 64, 64)], Parallel(), first=True, last=True)
    names = {o.name for o in g.ops}
    assert {"qkv_proj_act_quant", "o_proj_act_quant", "down_proj_act_quant"} <= names


def test_trace_frontend_counts_the_same_arithmetic():
    torch = pytest.importorskip("torch")
    from torch import nn
    from torch.nn import functional as F

    from infera.projection.core.projection.inference_projection.dllm.trace import trace_model

    spec = _tiny()
    h, nh, nkv, hd = spec.hidden_size, spec.num_attention_heads, spec.num_query_groups, 32

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(h, (nh + 2 * nkv) * hd, bias=False)
            self.o = nn.Linear(nh * hd, h, bias=False)
            self.up = nn.Linear(h, 2 * spec.ffn_hidden_size, bias=False)
            self.down = nn.Linear(spec.ffn_hidden_size, h, bias=False)
            self.n1, self.n2 = nn.RMSNorm(h), nn.RMSNorm(h)

        def forward(self, x, causal: bool):
            b, s, _ = x.shape
            q, k, v = self.qkv(self.n1(x)).split([nh * hd, nkv * hd, nkv * hd], -1)
            q = q.view(b, s, nh, hd).transpose(1, 2)
            k = k.view(b, s, nkv, hd).transpose(1, 2).repeat_interleave(nh // nkv, 1)
            v = v.view(b, s, nkv, hd).transpose(1, 2).repeat_interleave(nh // nkv, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            x = x + self.o(a.transpose(1, 2).reshape(b, s, nh * hd))
            gate, up = self.up(self.n2(x)).chunk(2, -1)
            return x + self.down(F.silu(gate) * up)

    class Model(nn.Module):
        def __init__(self, causal: bool) -> None:
            super().__init__()
            self.causal = causal
            self.emb = nn.Embedding(spec.vocab_size, h)
            self.layers = nn.ModuleList(Layer() for _ in range(spec.num_layers))
            self.norm = nn.RMSNorm(h)
            self.head = nn.Linear(h, spec.vocab_size, bias=False)

        def forward(self, ids):
            x = self.emb(ids)
            for layer in self.layers:
                x = layer(x, self.causal)
            return self.head(self.norm(x))

    batch, seq = 2, 64
    with torch.device("meta"):
        bidir, causal = Model(False).to(torch.bfloat16), Model(True).to(torch.bfloat16)
    ids = torch.zeros(batch, seq, dtype=torch.long, device="meta")
    tb = trace_model(bidir, ids)
    tc = trace_model(causal, ids)

    g = build_stage_graph(
        spec, [SeqShape(seq, seq, seq)] * batch, Parallel(), first=True, last=True
    )
    assert tb.flops(GEMM) == pytest.approx(_flops(g, GEMM), rel=1e-6)
    assert tb.flops(ATTENTION) == pytest.approx(_flops(g, ATTENTION), rel=1e-6)
    assert tb.flops(ATTENTION) == pytest.approx(2 * tc.flops(ATTENTION))


def _trace_one(module, x):
    from infera.projection.core.projection.inference_projection.dllm.trace import trace_model

    gemms = [o for o in trace_model(module, x).ops if o.kind == GEMM]
    assert len(gemms) == 1
    return gemms[0]


def test_convolution_lowers_to_its_implicit_gemm():
    torch = pytest.importorskip("torch")
    from torch.utils.flop_counter import FlopCounterMode

    with torch.device("meta"):
        conv = torch.nn.Conv2d(64, 128, 3, padding=1, bias=False).to(torch.bfloat16)
    x = torch.zeros(2, 64, 32, 32, dtype=torch.bfloat16, device="meta")
    op = _trace_one(conv, x)
    assert op.gemm == (2 * 32 * 32, 128, 64 * 9, 1)
    m, n, k, b = op.gemm
    assert op.flops == pytest.approx(2.0 * m * n * k * b)
    with FlopCounterMode(display=False) as fc:
        conv(x)
    assert op.flops == pytest.approx(fc.get_total_flops())
    # Real tensors, not an im2col buffer: input + weight + output.
    assert op.bytes_moved == pytest.approx(
        (2 * 64 * 32 * 32 + 128 * 64 * 9 + 2 * 128 * 32 * 32) * 2
    )


def test_grouped_strided_and_transposed_convolutions_lower_consistently():
    torch = pytest.importorskip("torch")
    from torch.utils.flop_counter import FlopCounterMode

    with torch.device("meta"):
        cases = [
            (torch.nn.Conv2d(64, 128, 3, stride=2, padding=1, groups=4), (2, 64, 32, 32), 4),
            (torch.nn.Conv1d(256, 512, 5, padding=2), (4, 256, 1000), 1),
            (torch.nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), (2, 128, 16, 16), 1),
            (torch.nn.ConvTranspose2d(128, 64, 4, stride=2, groups=2), (2, 128, 16, 16), 2),
            (torch.nn.Conv3d(32, 64, 3, padding=1), (1, 32, 8, 16, 16), 1),
        ]
    for module, shape, groups in cases:
        module = module.to(torch.bfloat16)
        x = torch.zeros(*shape, dtype=torch.bfloat16, device="meta")
        op = _trace_one(module, x)
        m, n, k, b = op.gemm
        assert b == groups
        with FlopCounterMode(display=False) as fc:
            module(x)
        # The lowered problem does exactly the convolution's arithmetic.
        assert 2.0 * m * n * k * b == pytest.approx(fc.get_total_flops())


def test_depthwise_convolution_stays_on_the_roofline():
    torch = pytest.importorskip("torch")
    with torch.device("meta"):
        conv = torch.nn.Conv1d(1024, 1024, 4, groups=1024, padding=3).to(torch.bfloat16)
    op = _trace_one(conv, torch.zeros(8, 1024, 2048, dtype=torch.bfloat16, device="meta"))
    assert op.gemm is None
    assert op.memory_bound(HW)


def test_gemm_pricer_replaces_the_roofline_for_lowered_ops():
    from dataclasses import replace

    from infera.projection.core.projection.inference_projection.dllm.ops import Op

    class _Pricer:
        def __init__(self) -> None:
            self.seen: list[tuple] = []

        def time_ms(self, gemm, dtype):
            self.seen.append((gemm, dtype))
            return 0.5

    pricer = _Pricer()
    hw = replace(HW, gemm_pricer=pricer)
    op = Op("conv", GEMM, flops=1e12, bytes_read=1e6, gemm=(4096, 128, 576, 1))
    assert op.time_ms(hw, launch=False) == pytest.approx(0.5)
    assert pricer.seen == [((4096, 128, 576, 1), "bf16")]
    # Without a problem, or without a pricer, the roofline still applies.
    plain = Op("conv", GEMM, flops=1e12, bytes_read=1e6)
    assert plain.time_ms(hw, launch=False) == pytest.approx(plain.compute_ms(HW))
    assert op.time_ms(HW, launch=False) == pytest.approx(op.compute_ms(HW))
    # The config frontend's GEMMs carry their problems too.
    g = build_stage_graph(_tiny(), [SeqShape(64, 64, 64)], Parallel(), first=True, last=True)
    assert {o.name: o for o in g.ops}["qkv_proj"].gemm == (64, (8 + 4) * 32, 256, 1)


def test_origami_prices_a_lowered_convolution():
    from infera.projection.core.projection.inference_projection.dllm.hardware import (
        OrigamiGemmPricer,
    )

    if not OrigamiGemmPricer("mi355x").available:
        pytest.skip("Origami is not installed")
    hw = HW.with_gemm_backend("origami")
    from infera.projection.core.projection.inference_projection.dllm.ops import Op

    op = Op("conv", GEMM, flops=2.0 * 2048 * 128 * 576, bytes_read=1e6, gemm=(2048, 128, 576, 1))
    t = op.time_ms(hw, launch=False)
    # Origami is capped at the roofline the ops use, so it can only be faster.
    assert 0 < t <= max(op.compute_ms(hw), op.memory_ms(hw)) * (1 + 1e-9)


def test_gemm_backend_option():
    from infera.projection.core.projection.inference_projection.dllm.config import from_dict
    from infera.projection.core.projection.inference_projection.dllm.hardware import (
        OrigamiGemmPricer,
    )

    assert from_dict({"hardware": {"gemm_backend": "roofline"}}).hw.gemm_pricer is None
    auto = from_dict({"hardware": {"gemm_backend": "auto"}}).hw.gemm_pricer
    assert (auto is not None) == OrigamiGemmPricer("mi355x").available
    with pytest.raises(ValueError, match="gemm_backend"):
        HW.with_gemm_backend("cutlass")
    if not OrigamiGemmPricer("mi355x").available:
        with pytest.raises(RuntimeError, match="Origami"):
            HW.with_gemm_backend("origami")


# -- workload -----------------------------------------------------------------


def test_schedule_has_the_requested_steps_and_blocks():
    wl = DLLMWorkload(prompt_len=128, gen_len=256, block_len=64, steps=32, cache="dual")
    sched = wl.schedule()
    assert len(sched) == 32
    assert [s.finishes_block for s in sched if s.finishes_block >= 0] == [0, 1, 2, 3]
    # Each block opens with a full refresh, then recomputes only itself.
    assert sched[0].shape.q == 384 and sched[1].shape.q == 64


def test_block_diffusion_prefills_the_prompt_once():
    wl = DLLMWorkload(prompt_len=128, gen_len=128, block_len=32, steps=16, cache="block")
    sched = wl.schedule()
    assert sched[0].shape == SeqShape(128, 128, 0)
    assert all(s.shape.q == 32 for s in sched[1:])
    assert sched[-1].shape.kv == 128 + 128


def test_active_logits_only_predict_masked_positions():
    wl = DLLMWorkload(prompt_len=64, gen_len=64, block_len=32, steps=8, logits="active")
    assert all(s.shape.logits <= 32 for s in wl.schedule())


def test_cache_modes_order_by_work():
    spec = load_model("llada_8B")
    model = DLLMStepModel(spec, Parallel(), HW)
    lat = {}
    for cache in ("none", "prefix", "dual", "block"):
        wl = DLLMWorkload(prompt_len=512, gen_len=256, block_len=32, steps=64, cache=cache)
        lat[cache] = simulate_static(model, wl, 4).latency_ms
    assert lat["none"] > lat["prefix"] > lat["dual"] > lat["block"]


def test_guidance_doubles_the_sequences():
    spec = _tiny()
    model = DLLMStepModel(spec, Parallel(), HW)
    plain = simulate_static(model, DLLMWorkload(prompt_len=64, gen_len=64, steps=8), 8)
    cfg = simulate_static(model, DLLMWorkload(prompt_len=64, gen_len=64, steps=8, cfg=True), 4)
    assert cfg.latency_ms == pytest.approx(plain.latency_ms)


# -- communication ------------------------------------------------------------


def test_activation_payload_is_tokens_hidden_precision():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(pp=2), HW, CommConfig("mxfp4", "all"))
    tokens = 4 * 1024
    hop = comm.pp_hop(tokens, cross_node=True)
    assert hop.wire_bytes == pytest.approx(tokens * spec.hidden_size * 0.53125)
    assert dtype_bytes("mxfp4") == 0.53125 and dtype_bytes("fp8") == 1.0


def test_inter_node_hop_is_slower_than_intra_node():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(tp=8, pp=2), HW)
    assert comm.pp_boundaries() == [True]
    assert DLLMCommModel(spec, Parallel(tp=2, pp=2), HW).pp_boundaries() == [False]
    intra = comm.pp_hop(8192, cross_node=False).total_ms
    cross = comm.pp_hop(8192, cross_node=True).total_ms
    # Both carry the measured per-hop engine overhead; the 67 MB transfer is
    # what differs, and over a 50 GB/s NIC it is over a millisecond more.
    assert cross - intra > 1.0


def test_quantization_wins_on_the_nic_and_loses_on_small_messages():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(tp=8, pp=2), HW)
    assert comm.tradeoff("pp_hop_cross", 8192, "fp8")["saving_ms"] > 0
    # A decode-sized all-reduce: the pack kernels cost more than the bytes saved.
    assert comm.tradeoff("tp_allreduce", 16, "fp8")["saving_ms"] < 0


def test_quant_scope_inter_node_leaves_intra_node_collectives_alone():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(tp=8), HW, CommConfig("fp8", "inter_node"))
    assert comm.collective("tp_allreduce", 4096).quant_ms == 0.0
    comm = DLLMCommModel(spec, Parallel(tp=8), HW, CommConfig("fp8", "all"))
    assert comm.collective("tp_allreduce", 4096).quant_ms > 0.0


def test_collectives_default_precision_is_unchanged():
    from infera.projection.core.projection.inference_projection.collectives import (
        InferenceCollectiveModel,
    )
    from infera.projection.core.projection.training_config import (
        InferenceCollectiveConfig,
        ModelConfig,
        ModelParallelConfig,
    )

    icm = InferenceCollectiveModel(
        ModelConfig(hidden_size=4096, moe_router_topk=2),
        ModelParallelConfig(
            tensor_model_parallel_size=8,
            pipeline_model_parallel_size=2,
            expert_model_parallel_size=8,
        ),
        InferenceCollectiveConfig(),
        num_nodes=2,
        gpus_per_node=8,
    )
    for tokens in (1, 64, 4096):
        assert icm.tp_allreduce_ms(4, tokens) == icm.tp_allreduce_ms(4, tokens, 2.0)
        assert icm.ep_a2a_ms(4, tokens) == icm.ep_a2a_ms(4, tokens, 2.0)
        assert icm.pp_p2p_ms(4, tokens) == icm.pp_p2p_ms(4, tokens, 2.0)
    assert icm.tp_allreduce_ms(4, 4096, 1.0) < icm.tp_allreduce_ms(4, 4096, 2.0)
    assert icm.pp_hop_ms(0, cross_node=True) == 0.0
    assert icm.pp_hop_ms(1e7, cross_node=True) > icm.pp_hop_ms(1e7, cross_node=False)


# -- overlap and contention ---------------------------------------------------


def test_overlap_limits():
    cfg = ContentionConfig()
    kw = {"wire_bytes": 1e8, "mem_fraction": 0.5, "hw": HW, "cfg": cfg}
    none = overlap(10.0, 4.0, overlap_fraction=0.0, **kw)
    assert none.total_ms == pytest.approx(14.0)
    ideal = overlap(10.0, 4.0, overlap_fraction=1.0, **{**kw, "cfg": ContentionConfig(False)})
    assert ideal.total_ms == pytest.approx(10.0)
    real = overlap(10.0, 4.0, overlap_fraction=1.0, **kw)
    assert 10.0 < real.total_ms < 14.0
    assert real.contention_ms == pytest.approx(real.total_ms - 10.0)
    comm_bound = overlap(2.0, 4.0, overlap_fraction=1.0, **{**kw, "cfg": ContentionConfig(False)})
    assert comm_bound.total_ms == pytest.approx(4.0)


def test_contention_hits_the_resource_the_collective_loads():
    kw = {"overlap_fraction": 1.0, "hw": HW, "cfg": ContentionConfig()}
    # 2 GB in 4 ms takes ~19% of HBM, more than RCCL's 6% of the CUs:
    # memory-bound kernels lose more.
    mem = overlap(10.0, 4.0, mem_fraction=1.0, wire_bytes=2e9, **kw)
    comp = overlap(10.0, 4.0, mem_fraction=0.0, wire_bytes=2e9, **kw)
    assert mem.total_ms > comp.total_ms
    # 200 MB over a NIC-speed 4 ms barely touches HBM; the CUs are the cost.
    mem = overlap(10.0, 4.0, mem_fraction=1.0, wire_bytes=2e8, **kw)
    comp = overlap(10.0, 4.0, mem_fraction=0.0, wire_bytes=2e8, **kw)
    assert comp.total_ms > mem.total_ms


def test_overlap_shortens_a_tensor_parallel_step():
    spec = load_model("llada_8B")
    seqs = [SeqShape(1024, 1024, 1024)] * 8
    sync = DLLMStepModel(spec, Parallel(tp=8), HW).step(seqs)
    over = DLLMStepModel(spec, Parallel(tp=8), HW, CommConfig(overlap=1.0)).step(seqs)
    assert over.total_ms < sync.total_ms
    assert over.as_dict()["contention_ms"] > 0


# -- pipeline -----------------------------------------------------------------


def test_pipeline_microbatches_fill_the_stages():
    spec = load_model("llada_8B")
    seqs = [SeqShape(512, 512, 512)] * 8
    one = DLLMStepModel(spec, Parallel(tp=8, pp=4, pp_microbatches=1), HW).step(seqs)
    four = DLLMStepModel(spec, Parallel(tp=8, pp=4, pp_microbatches=4), HW).step(seqs)
    assert four.microbatches == 4 and one.microbatches == 1
    assert four.total_ms < one.total_ms
    assert four.token_return_ms > 0


# -- memory -------------------------------------------------------------------


def test_logits_dominate_activation_memory():
    spec = load_model("llada_8B")
    wl = DLLMWorkload(prompt_len=512, gen_len=512, steps=64)
    rep = memory_report(spec, Parallel(), HW, wl, 8)
    # BF16 logits plus two FP32 copies over the whole canvas.
    assert rep.activations_gb == pytest.approx(8 * 1024 * 126464 * 10 / 1e9)
    assert rep.kv_cache_gb == 0.0


def test_max_batch_is_the_memory_boundary():
    spec = load_model("llada_8B")
    wl = DLLMWorkload(prompt_len=512, gen_len=512, steps=64, cache="dual")
    b = max_batch(spec, Parallel(), HW, wl)
    assert b > 1
    assert memory_report(spec, Parallel(), HW, wl, b).fits
    assert not memory_report(spec, Parallel(), HW, wl, b + 1).fits
    assert max_batch(spec, Parallel(tp=8), HW, wl) > b


# -- simulation ---------------------------------------------------------------


def test_serving_at_low_rate_matches_an_unloaded_request():
    spec = load_model("llada_8B")
    model = DLLMStepModel(spec, Parallel(tp=8), HW)
    wl = DLLMWorkload(prompt_len=256, gen_len=128, block_len=32, steps=32, cache="dual")
    alone = simulate_static(model, wl, 1).latency_ms
    res = simulate_serving(model, wl, rate_per_s=0.05, num_requests=40)
    assert res.latency_ms["p50"] == pytest.approx(alone, rel=0.05)
    assert not res.saturated


def test_serving_batches_up_under_load():
    spec = load_model("llada_8B")
    model = DLLMStepModel(spec, Parallel(tp=8), HW)
    wl = DLLMWorkload(prompt_len=256, gen_len=128, block_len=32, steps=32, cache="dual")
    low = simulate_serving(model, wl, rate_per_s=1, num_requests=100)
    high = simulate_serving(model, wl, rate_per_s=200, num_requests=300, batch_limit=8)
    assert high.mean_batch > low.mean_batch
    assert high.tokens_per_s > low.tokens_per_s
    assert high.saturated


def test_inferasim_backend_drops_the_projectors_comm():
    class _Comm:
        total_ms = 1.5

    class _FT:
        total_ms = 10.0
        comm = _Comm()

    class _Projector:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def _forward_times(self, batch, q_len, phase, kv_len):
            self.calls.append((batch, q_len, phase, kv_len))
            return _FT()

    proj = _Projector()
    backend = InferaSimBackend(proj)
    spec = load_model("llada_8B")
    model = DLLMStepModel(spec, Parallel(tp=8), HW, backend=backend)
    step = model.step([SeqShape(256, 512, 256), SeqShape(128, 512, 128)])
    assert proj.calls == [(2, 192, "refine", 512)]
    assert step.stages[0].compute_ms == pytest.approx(8.5)


# -- CLI ----------------------------------------------------------------------


def test_cli_runs_a_config(tmp_path, capsys):
    from infera.projection.cli import main

    cfg = tmp_path / "dllm.yaml"
    cfg.write_text(
        "model: llada_8B\n"
        "parallel: {tp: 8}\n"
        "workload: {prompt_len: 128, gen_len: 64, block_len: 32, steps: 16, cache: dual}\n"
        "run:\n  static_batches: [1, 4]\n  serving: {rates: [2], num_requests: 20}\n"
    )
    out = tmp_path / "out.json"
    assert main(["dllm", "--config", str(cfg), "--json", str(out)]) == 0
    res = json.loads(out.read_text())
    assert [s["batch"] for s in res["static"]] == [1, 4]
    assert res["serving"][0]["completed"] == 20
    assert "dLLM-Sim" in capsys.readouterr().out


def test_config_rejects_unknown_keys():
    from infera.projection.core.projection.inference_projection.dllm.config import from_dict

    with pytest.raises(ValueError, match="parallel"):
        from_dict({"parallel": {"tpp": 2}})
    with pytest.raises(ValueError, match="section"):
        from_dict({"paralel": {}})
