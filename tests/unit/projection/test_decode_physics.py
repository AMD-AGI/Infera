###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Physical bounds the decode cost model must not violate.

Each of these encodes a defect the measured TP ladder exposed. They are written
as inequalities against hardware limits rather than as expected values, so they
keep holding when the model gets more accurate but fail the moment a term goes
back to being unbounded.
"""

from __future__ import annotations

import pytest

from .conftest import DEFAULTS, project_spec

# MI355X: HBM3E, 8 TB/s peak. Nothing that streams bytes may imply more.
_MI355X_HBM_TBPS = 8.0


def _sdpa(**kw):
    from infera.projection.core.projection.simulation_backends.sdpa_simulator import (
        SDPASimulator,
    )

    return SDPASimulator(gpu_arch="mi355x").simulate_sdpa(**kw)


def test_sdpa_decode_cannot_beat_hbm_bandwidth():
    """A decode step must not read the KV cache faster than HBM allows.

    The tile model prices per-workgroup GEMMs on one CU and scales by wave
    count, which never bounds the result by device bandwidth. Unbounded, it
    returned 0.27 ms for 19.3 GB of gpt-oss KV -- about 71 TB/s.
    """
    batch, kv_len, kv_heads, head_dim = 256, 1536, 8, 64
    res = _sdpa(
        batch_size=batch, num_heads=64, seq_len=1, head_dim=head_dim,
        causal=False, dtype="bf16", seq_len_kv=kv_len,
        num_heads_kv=kv_heads, head_dim_v=head_dim,
    )
    kv_bytes = batch * kv_heads * kv_len * (head_dim * 2) * 2  # K and V, bf16
    implied_tbps = kv_bytes / (res.forward_time_ms * 1e-3) / 1e12
    assert implied_tbps <= _MI355X_HBM_TBPS, (
        f"SDPA implies {implied_tbps:.1f} TB/s on an {_MI355X_HBM_TBPS} TB/s part"
    )


def test_mla_kv_is_a_shared_latent_not_per_head():
    """MLA caches one latent for all heads; charging per head overstates it.

    DeepSeek's cache holds ``kv_lora_rank + rope`` per token, shared across
    heads and replicated (not sharded) across TP ranks. Deriving the footprint
    from head counts overstated it ~9x and put DeepSeek TPOT +114% at
    concurrency 256.
    """
    common = dict(batch_size=128, num_heads=16, seq_len=1, head_dim=192,
                  causal=False, dtype="bf16", seq_len_kv=2048,
                  num_heads_kv=16, head_dim_v=128)
    per_head = _sdpa(**common)
    latent = _sdpa(**common, kv_bytes_per_token=(512 + 64) * 1.0)
    assert latent.forward_time_ms < per_head.forward_time_ms, (
        "supplying MLA's real per-token cache size must cost less than the "
        "per-head footprint the head counts imply"
    )


def test_attention_shards_heads_so_it_scales_with_tp():
    """TP splits attention heads, so per-rank attention must fall with TP.

    Sharding the token axis instead left every rank streaming the whole
    Q/K/V/O weights, and modelled attention was near-flat from TP=1 to TP=8
    while the measured ladder shards strongly.
    """
    tp1 = project_spec(tp=1, concurrency=64)["decode_step_ms"]
    tp8 = project_spec(tp=8, concurrency=64)["decode_step_ms"]
    assert tp8 < tp1, "decode step must fall when the model is sharded wider"


def test_kernel_occupancy_is_additive_and_survives_graph_capture():
    """Occupancy adds to data movement rather than capping it.

    Graph replay removes host dispatch, not kernel execution, and the small
    latency-bound kernels of a decode layer run alongside the large data-bound
    ones. Modelled as ``max`` it vanished exactly at the batch sizes where the
    measurement still showed it.
    """
    from infera.projection.core.projection.training_config import InferenceRequestConfig

    req = InferenceRequestConfig(
        decode_kernel_occupancy_us=6.0, kernels_per_layer=12,
        cudagraph_mode="full",
    )
    occ = req.resolved_decode_occupancy_ms(num_layers=36)
    assert occ > 0.0, "graph capture must not zero the occupancy term"
    assert occ == pytest.approx((36 * 12 + 6) * 6.0 / 1000.0)

    # And the launch-latency floor, which models host dispatch, must still be
    # cancelled by capture -- the two terms are not the same thing.
    launch = InferenceRequestConfig(
        kernel_launch_latency_us=6.0, kernels_per_layer=12, cudagraph_mode="full",
    ).resolved_kernel_launch_floor_ms(num_layers=36)
    assert launch == 0.0


def test_kernel_occupancy_is_read_off_the_silicon_not_a_global_default():
    """One vendor's measured floor is not the other's.

    The per-kernel occupancy was solved on mi355x. Applied to gb300 it prices
    GLM-5.2's 1,800 decode kernels at 8.96 ms of fixed cost, where the measured
    gb300 deployments report 3.0-3.6 ms per output token at batch 1 -- so a
    single global constant cannot be a hardware figure for both. An architecture
    nobody has measured keeps the default rather than inheriting a neighbour's:
    the cost is real wherever it runs, and guessing its size from who made the
    part is what this table exists to stop.
    """
    from infera.projection.core.projection.training_config import (
        DEFAULT_DECODE_OCCUPANCY_US,
        resolve_decode_occupancy_us,
    )

    solved_on = resolve_decode_occupancy_us("mi355x")
    other = resolve_decode_occupancy_us("gb300")
    assert other < solved_on, "the two architectures do not share a floor"
    assert resolve_decode_occupancy_us("MI355X") == solved_on, "arch names are not case-sensitive"
    for unmeasured in (None, "", "h100", "something-new"):
        assert resolve_decode_occupancy_us(unmeasured) == DEFAULT_DECODE_OCCUPANCY_US


def test_the_projector_charges_the_floor_the_architecture_resolves_to():
    """The table has to reach the projection, not just be readable.

    An unset occupancy means "resolve from the GPU", so the resolved figure has
    to move the step: halving it has to take roughly half the fixed cost out at
    batch 1, where fixed cost is most of what is left. Asserted against an
    explicit override on one architecture rather than by comparing two, so the
    wiring is checked without needing a second part's roofline profile to exist.
    """
    resolved = project_spec(tp=8, concurrency=1, gpu_arch="mi355x")["decode_step_ms"]
    from infera.projection.core.projection.training_config import (
        resolve_decode_occupancy_us,
    )

    occ = resolve_decode_occupancy_us("mi355x")
    halved = project_spec(
        tp=8, concurrency=1, gpu_arch="mi355x", decode_kernel_occupancy_us=occ / 2.0
    )["decode_step_ms"]
    assert halved < resolved, (
        "the resolved per-architecture floor never reached the decode step "
        f"({resolved:.2f} ms unchanged when the occupancy was halved)"
    )


def test_occupancy_reaches_the_continuous_batching_path():
    """Every vLLM workload goes through continuous batching, not the static path.

    The term was charged only in ``_decode_step_latency_ms``, so the
    disaggregated path had it and the continuous one silently did not.
    """
    base = project_spec(tp=8, concurrency=64, decode_kernel_occupancy_us=0.0)
    with_occ = project_spec(tp=8, concurrency=64, decode_kernel_occupancy_us=6.0)
    assert with_occ["tpot_ms"] > base["tpot_ms"], (
        "occupancy must move TPOT on the continuous-batching path"
    )


def test_decode_all_reduce_matches_the_measured_custom_kernel():
    """The decode all-reduce must track what vLLM's own kernel actually costs.

    vLLM dispatches everything under 8 MB to its custom all-reduce, not RCCL, so
    the 26 us RCCL floor was the wrong kernel: it put a flat 1.872 ms into every
    gpt-oss decode step at TP>1, independent of batch *and* of TP, because the
    floor beat the bandwidth term even at a 1.5 MB message.

    Measured at hidden 2880:
    9.9 us at batch 1 (5.8 KB) and 25.1 us at batch 256 (1.5 MB) on 8 ranks.

    What the model charges is the *marginal* cost over per-kernel occupancy,
    since the all-reduce is one of the step's kernels and its occupancy is
    already counted there. So the measured standalone time is reproduced by
    adding the occupancy floor back, and that is what this checks.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        _INFER_AR_MEASURED_GBPS,
        _measured_intra_node_ar_us,
    )

    floor_us = _INFER_AR_MEASURED_GBPS[8][0]
    small = _measured_intra_node_ar_us(1 * 2880 * 2, gpus=8)
    large = _measured_intra_node_ar_us(256 * 2880 * 2, gpus=8)

    assert small + floor_us == pytest.approx(9.9, rel=0.30), (
        f"batch-1 all-reduce {small + floor_us:.1f} us"
    )
    assert large + floor_us == pytest.approx(25.1, rel=0.30), (
        f"batch-256 all-reduce {large + floor_us:.1f} us"
    )
    # The defect that started this was flatness, not the constant: a 256x larger
    # message has to cost meaningfully more, or batch cannot move the comm term.
    assert large > 2.0 * small, "all-reduce is flat in message size again"


def test_a_small_all_reduce_is_not_charged_its_occupancy_twice():
    """A decode step already pays per-kernel occupancy for every kernel it runs,
    the collective included. Charging the collective's own fitted floor on top
    made the step's fixed cost climb with TP -- 1.91 ms at TP=1 to 2.73 ms at
    TP=8 -- when the measured floor is flat at 2.72 ms across all four rungs,
    because sharding does not remove kernels.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        _measured_intra_node_ar_us,
    )

    tiny = _measured_intra_node_ar_us(1 * 2880 * 2, gpus=8)
    assert tiny < 1.0, (
        f"a 5.8 KB all-reduce should cost well under a microsecond of transfer "
        f"beyond the occupancy every kernel pays, got {tiny:.2f} us"
    )


def test_a_layer_costs_the_kernels_it_actually_runs():
    """The per-step floor is a kernel count times a hardware minimum, and only
    their product was measured. Getting the split wrong is invisible on the
    model it was fitted to and wrong everywhere else: DeepSeek-R1 splits its
    attention projections for MLA and runs a shared expert every step, so its
    layer issues meaningfully more kernels than gpt-oss's.
    """
    from types import SimpleNamespace

    from infera.projection.core.projection.training_config import (
        decode_kernels_per_layer,
    )

    gpt_oss = SimpleNamespace(num_experts=128, multi_latent_attention=False,
                              moe_shared_expert_intermediate_size=None)
    deepseek = SimpleNamespace(num_experts=256, multi_latent_attention=True,
                               moe_shared_expert_intermediate_size=2048)
    dense = SimpleNamespace(num_experts=0, multi_latent_attention=False,
                            moe_shared_expert_intermediate_size=None)

    n_gpt, n_ds, n_dense = (decode_kernels_per_layer(m)
                            for m in (gpt_oss, deepseek, dense))
    assert n_dense < n_gpt < n_ds, (
        "a dense layer runs fewest, MLA plus a shared expert the most"
    )
    # MLA adds five kernels over fused QKV; a shared expert adds three.
    assert n_ds - n_gpt == 8


def test_the_step_floor_still_comes_out_where_it_was_measured():
    """The count and the constant have to be the pair that was calibrated.

    4.98 us is not a per-kernel measurement. It is a measured 2.72 ms step
    floor divided by the 546 kernels gpt-oss issues, so the projector only
    reproduces that floor if it charges the constant over the same 546.
    Charging it over the elementwise kernels alone type-checks, keeps every
    gpt-oss test green, and quietly halves the floor to 1.28 ms -- which a
    tensor-parallel sweep then reports as a scaling error, because the missing
    cost is the part that does not shard.
    """
    from types import SimpleNamespace

    from infera.projection.core.projection.training_config import (
        InferenceRequestConfig,
        decode_kernels_per_layer,
    )

    gpt_oss = SimpleNamespace(num_experts=128, multi_latent_attention=False,
                              moe_shared_expert_intermediate_size=None)
    assert 36 * decode_kernels_per_layer(gpt_oss) + 6 == 546

    floor = InferenceRequestConfig(
        decode_kernel_occupancy_us=4.98,
    ).resolved_decode_occupancy_ms(36, decode_kernels_per_layer(gpt_oss))
    assert floor == pytest.approx(2.72, abs=0.01), (
        f"the projector charges a {floor:.2f} ms floor where 2.72 ms was measured"
    )


def test_measured_router_coverage_reshapes_the_expert_count():
    """Real tokens route in a correlated way, so a step touches fewer experts.

    Measured from gpt-oss-120b's own router weights: coverage falls to ~0.70 of
    the independent-routing prediction around batch 32-64 and returns toward 1
    at both ends. The same probe on uncorrelated inputs stays near 1, which is
    what makes this token correlation rather than expert-popularity skew -- and
    is why no marginal-skew law reproduces the shape.
    """
    from infera.projection.core.projection.module_profilers.moe_mlp import (
        _router_coverage,
    )

    curve = {1: 1.0, 8: 0.85, 32: 0.71, 64: 0.70, 256: 0.84}
    assert _router_coverage(curve, 32) == pytest.approx(0.71)
    # Between measured points, interpolated rather than snapped.
    mid = _router_coverage(curve, 16)
    assert 0.71 < mid < 0.85
    # Held flat outside the measured range instead of extrapolated off a cliff.
    assert _router_coverage(curve, 1) == pytest.approx(1.0)
    assert _router_coverage(curve, 4096) == pytest.approx(0.84)
    # Unmeasured models keep independent routing.
    assert _router_coverage(None, 32) == 1.0
    assert _router_coverage({}, 32) == 1.0


def test_small_expert_groups_do_not_saturate_bandwidth():
    """A few experts cannot saturate HBM; ~16 can.

    Measured on MI355X at M=1 (gpt-oss): 14% of peak for one expert, 46% for
    four, 85% by sixteen. The same sweep at fixed expert count across etp=1..8
    stays at 72-78%, so this is set by the size of the *group*, not by the bytes
    each rank reads -- which is why TP sharding must not be charged for it.
    """
    from infera.projection.core.projection.module_profilers.moe_mlp import (
        _GROUPED_GEMM_PLATEAU,
        _grouped_gemm_efficiency,
    )

    assert _grouped_gemm_efficiency(1) < 0.30, "one expert should be far off peak"
    assert _grouped_gemm_efficiency(4) == pytest.approx(0.46, abs=0.12)
    # Saturated by ~16, so the term stops mattering above small batch.
    assert _grouped_gemm_efficiency(16) > 0.95 * _GROUPED_GEMM_PLATEAU
    assert _grouped_gemm_efficiency(128) == pytest.approx(_GROUPED_GEMM_PLATEAU, rel=0.02)
    # Monotone: more streams can never read slower.
    vals = [_grouped_gemm_efficiency(n) for n in (1, 2, 4, 8, 16, 32)]
    assert vals == sorted(vals)


def test_realistic_router_imbalance_barely_moves_the_expert_count():
    """Routing skew is not a free knob for fixing decode error.

    Fitting the skew to the measured ladder finds a clean optimum, which is
    tempting and wrong: it sits at an expert load imbalance far beyond what a
    router trained with a balance loss runs at. At realistic imbalance the
    distinct-expert count barely moves, so a large gain from this knob means it
    is absorbing something else.
    """
    from infera.projection.core.projection.module_profilers.moe_mlp import (
        _expert_hit_fraction,
    )

    experts, topk, tokens = 128, 4, 16  # gpt-oss-120b at batch 16
    uniform = _expert_hit_fraction(experts, topk, tokens, skew=0.0)
    # s=0.3 is already a 3.1x max/mean load, more skew than a balanced router.
    realistic = _expert_hit_fraction(experts, topk, tokens, skew=0.3)

    assert realistic == pytest.approx(uniform, rel=0.10), (
        f"realistic skew moved the count {uniform:.3f} -> {realistic:.3f}; "
        "if this becomes large the knob is doing someone else's job"
    )
    # It must still be a real term: heavy skew has to reduce the count.
    assert _expert_hit_fraction(experts, topk, tokens, skew=1.0) < 0.8 * uniform


@pytest.mark.parametrize(
    "name, k, ffn, experts", [("deepseek", 7168, 2048, 103), ("gpt_oss", 2880, 2880, 111)]
)
def test_expert_gemm_keeps_near_ideal_relief_when_sharded(name, k, ffn, experts):
    """Sharding the FFN dimension of the grouped expert GEMM must stay near 1/etp.

    Measured on MI355X: the grouped GEMM holds 57-78% of peak bandwidth across
    etp=1..8 and delivers 6.85-7.54x of the ideal 8x. So the roofline's
    near-ideal relief is right, and the batch-256 TP under-prediction does *not*
    come from here -- a plausible-sounding efficiency penalty tuned in at this
    spot would be a fudge factor covering for a term that lives elsewhere.

    Timing one expert at a time instead suggests a large penalty, but that is a
    benchmark artifact: a lone [1 x k] x [k x n] call is latency-bound and reads
    ~10% of peak, which is not the kernel MoE decode actually issues.
    """
    from infera.projection.core.projection.simulation_backends.origami_backend import (
        OrigamiGEMMBackend,
    )

    backend = OrigamiGEMMBackend(gpu_arch="mi355x")
    if not backend.is_available():
        pytest.skip("origami not installed")

    def t(etp):
        return backend.simulate_gemm(
            m=1, n=max(1, ffn // etp), k=k, dtype="bf16", batch=experts
        ).forward_time_ms

    relief = t(1) / t(8)
    assert 5.5 <= relief <= 8.4, (
        f"{name} expert GEMM relief at etp=8 is {relief:.2f}x; "
        "measurement puts it at 6.85-7.54x of the ideal 8x"
    )

