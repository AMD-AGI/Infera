###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Expert parallelism repartitions a replica's ranks; it does not add to them.

The legality check used to charge a replica ``tp * pp * ep`` GPUs, which on a
single 8-GPU node made TP8/EP8 -- the shape ``--enable-expert-parallel`` actually
produces -- illegal, while the projector it feeds reports replica GPUs as TP×PP
and models EP8 on those same eight ranks quite happily. The search therefore only
ever saw EP>1 with TP shrunk to fit (TP1/EP8, TP2/EP4, TP4/EP2), all of them at
the small batch those seeds carry, and reported EP1 as a result when it was a
constraint. These tests pin the rule to TP×PP so the answer is earned.
"""

from __future__ import annotations

from infera.projection.agents.tuning_agent.inference_tuning import (
    InferenceTrialConfig,
    build_inference_seed_plan,
    derive_inference_legality,
    validate_inference,
)


class _Arch:
    workload_path = "unused.yaml"
    num_attention_heads = 128
    hidden_size = 8192
    num_layers = 64
    is_moe = True
    num_experts = 256


class _Cluster:
    num_nodes = 1
    gpus_per_node = 8
    gpu_arch = "mi355x"


class _Opt:
    input_len = 130000
    output_len = 900
    max_concurrency = None
    hbm_capacity_gb = 288.0
    memory_safety_margin = 0.10
    slo = {}
    objective = "total_throughput_tps_per_gpu"


def _legality():
    return derive_inference_legality(_Arch(), _Cluster())


def _cfg(**kw) -> InferenceTrialConfig:
    base = dict(tp=8, ep=1, batch_size=16, weight_dtype="bf16", kv_cache_dtype="bf16")
    base.update(kw)
    return InferenceTrialConfig(**base)


def test_full_tp_with_full_ep_fits_on_one_node():
    """TP8/EP8 on 8 GPUs is the deployed shape, not a 64-GPU request."""
    ok, why = validate_inference(_cfg(ep=8), _Arch(), _Cluster(), _legality())
    assert ok, why


def test_expert_parallelism_composes_with_attention_dp_at_full_width():
    """The pairing the measured MLA fleets run has to be expressible."""
    ok, why = validate_inference(_cfg(ep=8, attention_dp=8), _Arch(), _Cluster(), _legality())
    assert ok, why


def test_an_ep_degree_that_does_not_divide_the_ranks_is_rejected():
    """EP4 over a 2-rank replica describes no assignment of experts to ranks."""
    ok, why = validate_inference(_cfg(tp=2, ep=4), _Arch(), _Cluster(), _legality())
    assert not ok and "divide" in why


def test_tensor_parallelism_is_still_bounded_by_the_cluster():
    """Relaxing the EP term must not relax the one that counts real GPUs."""
    leg = _legality()
    ok, why = validate_inference(_cfg(tp=16, ep=1), _Arch(), _Cluster(), leg)
    assert not ok, "TP16 does not fit on 8 GPUs"


def test_the_seed_plan_tries_expert_parallelism_at_serving_batch():
    """Seeded at batch 16 in bf16 only, EP dies on memory and is never scored.

    The question is whether disjoint experts beat a TP all-reduce at the batch
    the winners run, so the plan has to put EP there.
    """
    plan = build_inference_seed_plan(_Arch(), _Cluster(), _Opt(), max_candidates=4096)
    ep_trials = [c for c in plan.candidates if c.ep > 1]
    assert ep_trials, "no seed trial turns expert parallelism on"
    assert any(c.tp == 8 and c.ep == 8 for c in ep_trials), (
        "EP is only offered with TP shrunk to make room for it"
    )
    assert any(c.batch_size >= 64 and c.weight_dtype == "fp4" for c in ep_trials), (
        "EP is only seeded at the small bf16 batch that always dies on memory"
    )
