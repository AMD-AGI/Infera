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

from .conftest import project_spec

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


def test_small_decode_collective_is_not_charged_rccl_latency():
    """vLLM dispatches small decode messages to its own all-reduce, not RCCL.

    The 26 us RCCL floor put a flat 1.872 ms into every gpt-oss decode step at
    TP>1 -- independent of batch and of TP, because the floor beat the
    bandwidth term even at a 1.5 MB message. The ladder bounds the real cost
    under ~8 us per all-reduce.
    """
    from infera.projection.core.projection.inference_projection import collectives

    assert collectives._INFER_AR_OVERHEAD_US <= 8.0, (
        "intra-node decode all-reduce floor exceeds what the TP ladder allows"
    )


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
