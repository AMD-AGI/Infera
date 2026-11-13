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
from infera.projection.core.projection.inference_projection.dllm.contention import penalty_ms
from infera.projection.core.projection.inference_projection.dllm.hardware import dtype_bytes
from infera.projection.core.projection.inference_projection.dllm.ops import (
    ATTENTION,
    ATTN_BLOCK,
    DENSE_MLP,
    GEMM,
    MOE_MLP,
)

# The structural tests run anywhere, so they ask for the roofline explicitly;
# the default, Origami, is exercised where it is installed.
HW = DLLMHardware.from_arch("mi355x", backend="roofline")


def _origami_hw() -> DLLMHardware:
    try:
        return DLLMHardware.from_arch("mi355x")
    except (ImportError, RuntimeError):
        pytest.skip("Origami is not installed")


def _moe() -> DLLMModelSpec:
    return DLLMModelSpec(
        name="tiny_moe",
        num_layers=4,
        hidden_size=1024,
        ffn_hidden_size=2048,
        num_attention_heads=16,
        num_query_groups=4,
        vocab_size=1000,
        num_experts=8,
        moe_router_topk=2,
        moe_layer_freq=2,
    )


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
    assert (two.tokens, two.n_dense, two.n_moe) == (64, spec.num_layers, 0)


def test_attention_is_one_flash_problem_per_sequence_shape():
    spec = _tiny()
    seqs = [SeqShape(64, 64, 64), SeqShape(64, 64, 64), SeqShape(32, 96, 32)]
    g = build_stage_graph(spec, seqs, Parallel(tp=2), first=True, last=True)
    attn = sorted(o.attn for o in g.ops if o.kind == ATTENTION)
    heads, kv_heads = spec.num_attention_heads // 2, spec.num_query_groups // 2
    # Bidirectional: no causal mask, whatever the cache mode.
    assert attn == [
        (1, heads, 32, 96, spec.head_dim, kv_heads, False),
        (2, heads, 64, 64, spec.head_dim, kv_heads, False),
    ]
    assert all(o.group == ATTN_BLOCK for o in g.ops if o.kind == ATTENTION)


def test_ops_are_grouped_by_the_layer_type_they_overlap_with():
    g = build_stage_graph(_moe(), [SeqShape(64, 64, 64)], Parallel(), first=True, last=True)
    groups = {o.name: o.group for o in g.ops}
    assert groups["down_proj"] == DENSE_MLP and groups["expert_down"] == MOE_MLP
    assert groups["o_proj"] == ATTN_BLOCK and groups["lm_head"] == "once"
    assert (g.n_dense, g.n_moe) == (2, 2)


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


class _FakePricer:
    def __init__(self) -> None:
        self.seen: list[tuple] = []

    def gemm_ms(self, gemm, dtype):
        self.seen.append(("gemm", gemm, dtype))
        return 0.5

    def attention_ms(self, attn, dtype, kv_bytes_per_token):
        self.seen.append(("attn", attn, dtype, kv_bytes_per_token))
        return 0.25


def test_pricer_replaces_the_roofline_for_gemm_and_attention():
    from dataclasses import replace

    from infera.projection.core.projection.inference_projection.dllm.ops import Op

    pricer = _FakePricer()
    hw = replace(HW, pricer=pricer)
    assert hw.backend == "origami" and HW.backend == "roofline"
    op = Op("conv", GEMM, flops=1e12, bytes_read=1e6, gemm=(4096, 128, 576, 1))
    assert op.time_ms(hw, launch=False) == pytest.approx(0.5)
    shape = (2, 8, 64, 64, 32, 2, False)
    attn = Op("attention", ATTENTION, flops=1e9, attn=shape, kv_bytes_per_token=256.0)
    assert attn.time_ms(hw, launch=False) == pytest.approx(0.25)
    assert pricer.seen == [
        ("gemm", (4096, 128, 576, 1), "bf16"),
        ("attn", shape, "bf16", 256.0),
    ]
    # Without a problem, or without a pricer, the roofline still applies.
    plain = Op("conv", GEMM, flops=1e12, bytes_read=1e6)
    assert plain.time_ms(hw, launch=False) == pytest.approx(plain.compute_ms(HW))
    assert op.time_ms(HW, launch=False) == pytest.approx(op.compute_ms(HW))
    # The config frontend's GEMMs and attention carry their problems.
    g = build_stage_graph(_tiny(), [SeqShape(64, 64, 64)], Parallel(), first=True, last=True)
    assert {o.name: o for o in g.ops}["qkv_proj"].gemm == (64, (8 + 4) * 32, 256, 1)
    step = DLLMStepModel(_tiny(), Parallel(), hw).step([SeqShape(64, 64, 64)])
    assert step.total_ms > 0
    assert any(s[0] == "attn" and s[1][2:4] == (64, 64) for s in pricer.seen)


def test_origami_is_the_default_and_its_absence_is_an_error():
    from infera.projection.core.projection.inference_projection.dllm.config import from_dict

    assert from_dict({"hardware": {"backend": "roofline"}}).hw.backend == "roofline"
    with pytest.raises(ValueError, match="backend"):
        HW.with_backend("cutlass")
    with pytest.raises(ValueError, match="unknown hardware"):
        DLLMHardware.from_arch("mi355x", backend="roofline", pricer=object())
    try:
        import origami  # noqa: F401
    except ImportError:
        # The projector's rule: no Origami is an error, not a quiet roofline.
        with pytest.raises(RuntimeError, match="[Oo]rigami"):
            DLLMHardware.from_arch("mi355x")
        with pytest.raises(RuntimeError, match="[Oo]rigami"):
            from_dict({})
    else:
        assert DLLMHardware.from_arch("mi355x").backend == "origami"


def test_origami_prices_a_lowered_convolution_and_attention():
    hw = _origami_hw()
    from infera.projection.core.projection.inference_projection.dllm.ops import Op

    m, n, k = 2048, 128, 576
    op = Op(
        "conv",
        GEMM,
        flops=2.0 * m * n * k,
        bytes_read=(m * k + k * n) * 2,
        bytes_written=m * n * 2,
        gemm=(m, n, k, 1),
    )
    t = op.time_ms(hw, launch=False)
    # Origami is capped at the roofline the ops use, so it can only be faster.
    assert 0 < t <= max(op.compute_ms(hw), op.memory_ms(hw)) * (1 + 1e-9)
    shape = (4, 32, 1024, 1024, 128, 8, False)
    a = Op("attention", ATTENTION, flops=4.0 * 4 * 1024 * 1024 * 32 * 128, attn=shape)
    causal = Op("attention", ATTENTION, attn=(*shape[:-1], True))
    assert a.time_ms(hw, launch=False) > causal.time_ms(hw, launch=False) > 0


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


def _projector_comm(spec, par, gpn=8):
    """The collective model exactly as the autoregressive projector builds it."""
    from infera.projection.core.projection.inference_projection.collectives import (
        InferenceCollectiveModel,
    )
    from infera.projection.core.projection.training_config import (
        INTERCONNECT_PROFILES,
        InferenceCollectiveConfig,
        ModelConfig,
        ModelParallelConfig,
    )

    cc = InferenceCollectiveConfig()
    cc.interconnect = INTERCONNECT_PROFILES.get("mi355x")
    mp = ModelParallelConfig(
        tensor_model_parallel_size=par.tp,
        pipeline_model_parallel_size=par.pp,
        expert_model_parallel_size=par.ep,
    )
    mc = ModelConfig(
        hidden_size=spec.hidden_size,
        num_layers=spec.num_layers,
        num_attention_heads=spec.num_attention_heads,
        kv_channels=spec.head_dim,
        moe_router_topk=spec.moe_router_topk,
    )
    nodes = max(1, -(-par.tp * par.pp // gpn))
    return InferenceCollectiveModel(mc, mp, cc, num_nodes=nodes, gpus_per_node=gpn)


@pytest.mark.parametrize("par", [Parallel(tp=8), Parallel(tp=8, pp=2), Parallel(tp=4, pp=4)])
def test_dense_collectives_are_the_projectors(par):
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, par, HW)
    icm = _projector_comm(spec, par)
    for tokens in (64, 4096):
        # Both reductions of a dense layer: the projector's ``new_tp_ar``.
        assert comm.layer("dense", tokens).wire_ms == pytest.approx(icm.tp_allreduce_ms(1, tokens))
        assert comm.layer("dense", tokens).quant_ms == 0.0
        assert comm.pipeline(tokens).wire_ms == pytest.approx(icm.pp_p2p_ms(1, tokens))


@pytest.mark.parametrize("ep", [1, 2, 8])
def test_moe_collectives_use_the_projectors_reduction_count(ep):
    spec = _moe()
    par = Parallel(tp=8, ep=ep)
    comm = DLLMCommModel(spec, par, HW)
    icm = _projector_comm(spec, par)
    tokens = 1024
    ar = icm.tp_allreduce_ms(1, tokens)
    # tp // ep > 1 keeps the post-expert reduction; EP == TP drops it for the
    # all-to-all's combine (``_moe_tp_allreduce_count``).
    expected_ar = ar if ep < 8 else ar / 2
    a2a = icm.ep_a2a_ms(1, tokens) if ep > 1 else 0.0
    c = comm.layer("moe", tokens)
    assert c.wire_ms - c.a2a_ms == pytest.approx(expected_ar)
    assert c.a2a_ms == pytest.approx(a2a)


def test_overlap_is_the_projectors():
    from infera.projection.core.projection.inference_projection.performance import overlap_keep

    spec = load_model("llada_8B")
    for ceiling in (0.0, 0.5, 1.0):
        comm = DLLMCommModel(spec, Parallel(tp=8), HW, CommConfig(overlap=ceiling))
        assert comm.collective_config.prefill_overlap == ceiling
        for compute in (0.1, 2.0, 50.0):
            assert comm.keep(1.0, compute) == overlap_keep(ceiling, 1.0, compute)
    assert overlap_keep(0.0, 1.0, 5.0) == 1.0
    assert overlap_keep(1.0, 1.0, 5.0) == 0.0
    assert overlap_keep(1.0, 4.0, 1.0) == pytest.approx(0.75)
    assert overlap_keep(0.5, 1.0, None) == pytest.approx(0.5)


def test_step_charges_the_projectors_exposed_comm():
    spec = load_model("llada_8B")
    par = Parallel(tp=8)
    seqs = [SeqShape(1024, 1024, 1024)] * 2
    icm = _projector_comm(spec, par)
    step = DLLMStepModel(spec, par, HW).step(seqs)
    st = step.stages[0]
    # Synchronous by default: every layer's two reductions are exposed.
    assert st.exposed_comm_ms == pytest.approx(spec.num_layers * icm.tp_allreduce_ms(1, 2048))
    assert step.total_ms == pytest.approx(st.compute_ms + st.exposed_comm_ms)


def test_activation_payload_is_tokens_hidden_precision():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(pp=2), HW, CommConfig("mxfp4", "all"))
    tokens = 4 * 1024
    hop = comm.pipeline(tokens)
    assert hop.wire_bytes == pytest.approx(tokens * spec.hidden_size * 0.53125)
    assert dtype_bytes("mxfp4") == 0.53125 and dtype_bytes("fp8") == 1.0


def test_inter_node_pipeline_is_slower_than_intra_node():
    spec = load_model("llada_8B")
    cross = DLLMCommModel(spec, Parallel(tp=8, pp=2), HW)
    intra = DLLMCommModel(spec, Parallel(tp=4, pp=2), HW)
    assert cross.pp_crosses_node and not intra.pp_crosses_node
    # Both carry the measured per-hop engine overhead; the 67 MB transfer over
    # a 50 GB/s NIC is what differs.
    assert cross.pipeline(8192).wire_ms - intra.pipeline(8192).wire_ms > 1.0


def test_quantization_wins_on_the_nic_and_loses_on_small_messages():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(tp=8, pp=2), HW)
    assert comm.tradeoff("pipeline", 8192, "fp8")["saving_ms"] > 0
    # A decode-sized all-reduce: the pack kernels cost more than the bytes saved.
    assert comm.tradeoff("dense", 16, "fp8")["saving_ms"] < 0


def test_quant_scope_inter_node_leaves_intra_node_collectives_alone():
    spec = load_model("llada_8B")
    comm = DLLMCommModel(spec, Parallel(tp=8), HW, CommConfig("fp8", "inter_node"))
    assert comm.layer("dense", 4096).quant_ms == 0.0
    comm = DLLMCommModel(spec, Parallel(tp=8), HW, CommConfig("fp8", "all"))
    assert comm.layer("dense", 4096).quant_ms > 0.0


def test_collectives_default_precision_is_unchanged():
    icm = _projector_comm(_moe(), Parallel(tp=8, pp=2, ep=8))
    for tokens in (1, 64, 4096):
        assert icm.tp_allreduce_ms(4, tokens) == icm.tp_allreduce_ms(4, tokens, 2.0)
        assert icm.ep_a2a_ms(4, tokens) == icm.ep_a2a_ms(4, tokens, 2.0)
        assert icm.pp_p2p_ms(4, tokens) == icm.pp_p2p_ms(4, tokens, 2.0)
    assert icm.tp_allreduce_ms(4, 4096, 1.0) < icm.tp_allreduce_ms(4, 4096, 2.0)


# -- overlap and contention ---------------------------------------------------


def test_contention_is_off_by_default_and_costs_only_the_hidden_part():
    kw = {"wire_bytes": 1e8, "mem_fraction": 0.5, "hw": HW}
    assert penalty_ms(4.0, 0.0, cfg=ContentionConfig(), **kw) == 0.0
    on = ContentionConfig(enabled=True)
    assert penalty_ms(4.0, 1.0, cfg=on, **kw) == 0.0
    full = penalty_ms(4.0, 0.0, cfg=on, **kw)
    half = penalty_ms(4.0, 0.5, cfg=on, **kw)
    assert 0 < half < full < 4.0
    assert half == pytest.approx(full / 2)


def test_contention_hits_the_resource_the_collective_loads():
    kw = {"hw": HW, "cfg": ContentionConfig(enabled=True)}
    # 2 GB in 4 ms takes ~19% of HBM, more than RCCL's 6% of the CUs:
    # memory-bound kernels lose more.
    mem = penalty_ms(4.0, 0.0, mem_fraction=1.0, wire_bytes=2e9, **kw)
    comp = penalty_ms(4.0, 0.0, mem_fraction=0.0, wire_bytes=2e9, **kw)
    assert mem > comp
    # 200 MB over a NIC-speed 4 ms barely touches HBM; the CUs are the cost.
    mem = penalty_ms(4.0, 0.0, mem_fraction=1.0, wire_bytes=2e8, **kw)
    comp = penalty_ms(4.0, 0.0, mem_fraction=0.0, wire_bytes=2e8, **kw)
    assert comp > mem


def test_overlap_shortens_a_tensor_parallel_step():
    spec = load_model("llada_8B")
    seqs = [SeqShape(1024, 1024, 1024)] * 8
    sync = DLLMStepModel(spec, Parallel(tp=8), HW).step(seqs)
    over = DLLMStepModel(spec, Parallel(tp=8), HW, CommConfig(overlap=1.0)).step(seqs)
    assert over.total_ms < sync.total_ms
    assert over.as_dict()["contention_ms"] == 0.0
    real = DLLMStepModel(
        spec, Parallel(tp=8), HW, CommConfig(overlap=1.0), ContentionConfig(enabled=True)
    ).step(seqs)
    assert over.total_ms < real.total_ms < sync.total_ms
    assert real.as_dict()["contention_ms"] > 0


# -- pipeline -----------------------------------------------------------------


def test_pipeline_microbatches_fill_the_stages():
    spec = load_model("llada_8B")
    seqs = [SeqShape(512, 512, 512)] * 8
    one = DLLMStepModel(spec, Parallel(tp=8, pp=4, pp_microbatches=1), HW).step(seqs)
    four = DLLMStepModel(spec, Parallel(tp=8, pp=4, pp_microbatches=4), HW).step(seqs)
    assert four.microbatches == 4 and one.microbatches == 1
    assert four.total_ms < one.total_ms
    icm = _projector_comm(spec, Parallel(tp=8, pp=4))
    assert four.pp_p2p_ms == pytest.approx(icm.pp_p2p_ms(1, 2 * 512))


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
    from types import SimpleNamespace

    ft = SimpleNamespace(
        total_ms=10.0,
        layers_ms=8.0,
        dense_layer_ms=0.25,
        moe_layer_ms=0.0,
        comm=SimpleNamespace(tp_allreduce_ms=1.0, ep_a2a_ms=0.0, dcp_ms=0.0, pp_p2p_ms=0.5),
    )

    class _Projector:
        def __init__(self) -> None:
            self.calls: list[tuple] = []

        def _forward_times(self, batch, q_len, phase, kv_len):
            self.calls.append((batch, q_len, phase, kv_len))
            return ft

    proj = _Projector()
    backend = InferaSimBackend(proj)
    spec = load_model("llada_8B")
    model = DLLMStepModel(spec, Parallel(tp=8), HW, backend=backend)
    step = model.step([SeqShape(256, 512, 256), SeqShape(128, 512, 128)])
    assert proj.calls == [(2, 192, "refine", 512)]
    # Layers less their collectives, plus embedding and head: 7 + 1.5.
    assert step.stages[0].compute_ms == pytest.approx(8.5)


# -- CLI ----------------------------------------------------------------------


def test_cli_runs_a_config(tmp_path, capsys):
    from infera.projection.cli import main

    cfg = tmp_path / "dllm.yaml"
    cfg.write_text(
        "model: llada_8B\n"
        "hardware: {backend: roofline}\n"
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


# -- anchored mode ------------------------------------------------------------


def _tiny_lm():
    torch = pytest.importorskip("torch")
    spec = _tiny()

    class LM(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.emb = torch.nn.Embedding(spec.vocab_size, spec.hidden_size)
            self.mlp = torch.nn.Linear(spec.hidden_size, spec.hidden_size)
            self.head = torch.nn.Linear(spec.hidden_size, spec.vocab_size)

        def forward(self, ids):
            return self.head(torch.relu(self.mlp(self.emb(ids))))

    return spec, LM().eval()


def test_anchor_harvest_round_trips_and_transports(tmp_path):
    from infera.projection.core.projection.inference_projection.dllm.anchor import (
        AnchoredStepModel,
        harvest,
        load_anchor,
        save_anchor,
        target_regime,
    )

    spec, lm = _tiny_lm()
    anchor = harvest(
        lm,
        [(1, 32), (2, 64)],
        regime=target_regime(spec, attention_backend="sdpa", graph_mode="eager"),
        vocab_size=spec.vocab_size,
        warmup=1,
        iters=2,
    )
    path = tmp_path / "a.json"
    save_anchor(anchor, str(path))
    back = load_anchor(str(path))
    assert back.as_dict() == anchor.as_dict()
    assert [(p.batch, p.q, p.kv, p.logits) for p in back.points] == [
        (1, 32, 32, 32),
        (2, 64, 64, 64),
    ]
    assert all(p.step_ms > 0 for p in back.points)

    model = AnchoredStepModel(back, spec, Parallel(), HW)
    # At a measured point on the measured layout the projection is the measurement.
    for p in back.points:
        assert model.step(p.seqs()).total_ms == pytest.approx(p.step_ms)
    # Elsewhere it is that measurement times the simulator's ratio.
    target = [SeqShape(64, 64, 64)] * 4
    sim = DLLMStepModel(spec, Parallel(), HW)
    near = back.nearest(target)
    expected = near.step_ms * sim.step(target).total_ms / sim.step(near.seqs()).total_ms
    assert model.step(target).total_ms == pytest.approx(expected)
    # A new layout is priced against the anchor's own layout, not the target's.
    tp2 = AnchoredStepModel(back, spec, Parallel(tp=2), HW)
    assert tp2.reference is not tp2.target
    assert tp2.step(near.seqs()).total_ms == pytest.approx(
        near.step_ms * tp2.target.step(near.seqs()).total_ms / sim.step(near.seqs()).total_ms
    )


def test_anchor_refuses_another_regime(tmp_path):
    from infera.projection.core.projection.inference_projection.dllm.anchor import (
        AnchoredStepModel,
        AnchorMismatch,
        AnchorPoint,
        DLLMAnchor,
        find_anchor,
        save_anchor,
        target_regime,
    )

    spec = _tiny()
    a = DLLMAnchor(
        regime=target_regime(spec, attention_backend="sdpa"),
        points=[AnchorPoint(1, 64, 64, 64, 3.0)],
    )
    fp8 = DLLMModelSpec(**{**_tiny().__dict__, "weight_dtype": "fp8"})
    with pytest.raises(AnchorMismatch, match="weight_dtype"):
        AnchoredStepModel(a, fp8, Parallel(), HW)
    with pytest.raises(AnchorMismatch, match="attention_backend"):
        AnchoredStepModel(
            a, spec, Parallel(), HW, regime=target_regime(spec, attention_backend="triton")
        )
    # A target that does not state a key does not filter on it.
    AnchoredStepModel(a, spec, Parallel(), HW)

    save_anchor(a, str(tmp_path / "tiny.json"))
    assert find_anchor(str(tmp_path), target_regime(spec)).points[0].step_ms == 3.0
    with pytest.raises(AnchorMismatch, match="tiny.json: weight_dtype"):
        find_anchor(str(tmp_path), target_regime(fp8))


def test_cli_projects_from_an_anchor(tmp_path):
    from infera.projection.cli import main
    from infera.projection.core.projection.inference_projection.dllm.anchor import (
        AnchorPoint,
        DLLMAnchor,
        save_anchor,
        target_regime,
    )

    spec = load_model("llada_8B")
    store = tmp_path / "store"
    save_anchor(
        DLLMAnchor(
            regime=target_regime(spec),
            points=[AnchorPoint(1, 160, 160, 160, 50.0), AnchorPoint(4, 160, 160, 160, 120.0)],
        ),
        str(store / "llada.json"),
    )
    cfg = tmp_path / "dllm.yaml"
    cfg.write_text(
        "model: llada_8B\n"
        "workload: {prompt_len: 128, gen_len: 32, block_len: 32, steps: 8, cache: none}\n"
        "run:\n  static_batches: [1]\n"
    )
    out = tmp_path / "out.json"
    argv = ["dllm", "--config", str(cfg), "--backend", "roofline", "--json", str(out)]
    assert main([*argv, "--anchor-store", str(store)]) == 0
    res = json.loads(out.read_text())
    assert res["anchor"].endswith("llada.json")
    # Every step of a cacheless schedule is the measured 160-position forward.
    assert res["static"][0]["mean_step_ms"] == pytest.approx(50.0)


# -- runtime ------------------------------------------------------------------


def test_hf_eager_runtime_is_the_reference_implementations_kernels():
    from infera.projection.core.projection.inference_projection.dllm.runtime import RUNTIMES

    spec = _tiny()
    seqs = [SeqShape(64, 64, 64)] * 2
    engine = build_stage_graph(spec, seqs, Parallel(), first=True, last=True)
    eager = build_stage_graph(
        spec, seqs, Parallel(), first=True, last=True, runtime=RUNTIMES["hf_eager"]
    )
    names = {o.name for o in eager.ops}
    # Separate projections, the same arithmetic.
    assert {"q_proj", "k_proj", "v_proj", "up_gate_proj.gate", "up_gate_proj.up"} <= names
    assert "qkv_proj" not in names
    assert _flops(eager, GEMM) == pytest.approx(_flops(engine, GEMM), rel=1e-6)
    by = {o.name: o for o in eager.ops}
    assert by["k_proj"].gemm[1] == spec.num_query_groups * spec.head_dim
    # The element-wise chain moves more bytes and launches its kernels.
    e_norm, g_norm = by["attn_norm"], {o.name: o for o in engine.ops}["attn_norm"]
    rt = RUNTIMES["hf_eager"]
    assert e_norm.bytes_moved == pytest.approx(g_norm.bytes_moved * rt.elementwise_traffic)
    chain = [o for o in eager.ops if o.overhead_us is not None and o.name != "unmask_select"]
    per_layer = sum(o.overhead_us * o.count for o in chain) / spec.num_layers
    assert per_layer == pytest.approx(rt.elementwise_kernels_per_layer * rt.kernel_floor_us)
    # The FP32 sampler: 18 bytes per logit, not two bf16 passes.
    assert by["unmask_select"].bytes_read == pytest.approx(
        {o.name: o for o in engine.ops}["unmask_select"].bytes_read * 18 / 4
    )
    slow = DLLMStepModel(spec, Parallel(), HW, runtime="hf_eager").step(seqs).total_ms
    assert slow > DLLMStepModel(spec, Parallel(), HW).step(seqs).total_ms


def test_runtime_config_and_regime():
    from infera.projection.core.projection.inference_projection.dllm.config import from_dict

    run = from_dict({"hardware": {"backend": "roofline"}, "runtime": "hf_eager"})
    assert run.runtime.name == "hf_eager" and not run.runtime.fused_projections
    custom = from_dict(
        {"hardware": {"backend": "roofline"}, "runtime": {"preset": "hf_eager", "sampler": "fused"}}
    )
    assert custom.runtime.sampler == "fused" and custom.runtime.elementwise_traffic > 1
    with pytest.raises(ValueError, match="runtime"):
        from_dict({"hardware": {"backend": "roofline"}, "runtime": "vllm"})


def test_clock_override_scales_the_roofline():
    slow = DLLMHardware.from_arch("mi325x", backend="roofline", gpu_clock_mhz=1050)
    fast = DLLMHardware.from_arch("mi325x", backend="roofline")
    assert slow.peak_flops("bf16") == pytest.approx(fast.peak_flops("bf16") / 2)
    assert slow.mem_bytes_per_s() == fast.mem_bytes_per_s()


# -- interconnect ---------------------------------------------------------------


def test_mi325x_is_an_xgmi_mesh():
    """A TP2 pair drives one link of seven: the same bytes take ~3.8x a TP8 ring."""
    spec = load_model("llada_8B")
    hw = DLLMHardware.from_arch("mi325x", backend="roofline")
    tokens = 65536
    ar = {
        tp: DLLMCommModel(spec, Parallel(tp=tp), hw)._icm.tp_allreduce_ms(1, tokens, 2.0) / 2
        for tp in (2, 4, 8)
    }
    # RCCL on 8x MI325X, 512 MB: 11.3, 5.8, 3.0 ms.
    assert ar[2] == pytest.approx(11.3, rel=0.1)
    assert ar[4] == pytest.approx(5.8, rel=0.1)
    assert ar[8] == pytest.approx(3.0, rel=0.1)
    # The small-message fit, measured on a switched node, cannot beat one link.
    small = DLLMCommModel(spec, Parallel(tp=2), hw)._icm.tp_allreduce_ms(1, 1024, 2.0) / 2
    assert small >= 1024 * 4096 * 2 / (448e9 * 0.7 / 7) * 1e3 * 0.99
    # The switched mi355x node keeps its measured fit.
    sw = DLLMHardware.from_arch("mi355x", backend="roofline")
    t2 = DLLMCommModel(spec, Parallel(tp=2), sw)._icm.tp_allreduce_ms(1, 1024, 2.0) / 2
    assert t2 < small


def test_eager_pack_costs_what_was_measured():
    spec = load_model("llada_8B")
    hw = DLLMHardware.from_arch("mi325x", backend="roofline")
    fused = DLLMCommModel(spec, Parallel(tp=8), hw, CommConfig("fp8", "all"))
    eager = DLLMCommModel(spec, Parallel(tp=8), hw, CommConfig("fp8", "all", pack="eager"))
    f, e = fused.layer("dense", 65536, "fp8"), eager.layer("dense", 65536, "fp8")
    assert f.wire_ms == e.wire_ms
    # 5.8 ms of eager pack kernels per 512 MB FP8 all-reduce, two per layer.
    assert e.quant_ms / 2 == pytest.approx(5.8, rel=0.1)
    with pytest.raises(ValueError, match="pack"):
        CommConfig(pack="triton")
