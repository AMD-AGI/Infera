###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The serving search has to be able to ask for variable request lengths.

The projector has offered per-request heterogeneity for a while -- a uniform
spread via ``--des-range-ratio``, a replayed ``(arrival, isl, osl)`` workload via
``--des-workload-file`` -- but the tuning agent carried only ``request_rate`` and
``arrival_model``, so every trial was priced at a single ISL/OSL point. No
measured trace looks like that: the MI355X agentic runs span 103k-248k input, and
a point estimate cannot express long requests holding KV reservations while short
ones cycle through.
"""

from __future__ import annotations

from infera.projection.agents.tuning_agent.inference_tuning import (
    InferenceTrialConfig,
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


def _cfg(**kw) -> InferenceTrialConfig:
    base = dict(tp=8, ep=1, batch_size=16, weight_dtype="bf16", kv_cache_dtype="bf16")
    base.update(kw)
    return InferenceTrialConfig(**base)


def test_homogeneous_is_still_the_default():
    """A trial that does not ask for spread must project as it always did."""
    cfg = InferenceTrialConfig()
    assert cfg.des_range_ratio == 1.0
    assert cfg.des_workload_file is None
    assert cfg.des_num_requests == 0


def test_a_spread_is_expressible():
    leg = derive_inference_legality(_Arch(), _Cluster())
    ok, why = validate_inference(_cfg(des_range_ratio=0.6), _Arch(), _Cluster(), leg)
    assert ok, why


def test_a_nonsensical_spread_is_rejected_rather_than_clamped():
    """0 means zero-length prompts and >1 means longer than the configured ISL."""
    leg = derive_inference_legality(_Arch(), _Cluster())
    for bad in (0.0, -0.5, 1.5):
        ok, why = validate_inference(
            _cfg(des_range_ratio=bad), _Arch(), _Cluster(), leg
        )
        assert not ok and "des_range_ratio" in why, bad


def test_the_axes_reach_the_projector_command():
    """A knob the evaluator does not emit is a knob the search cannot use."""
    from infera.projection.agents.tuning_agent import evaluator as ev

    src = ev._build_inference_cmd.__code__.co_consts
    flat = " ".join(str(c) for c in src)
    for flag in ("--des-range-ratio", "--des-workload-file", "--des-num-requests"):
        assert flag in flat, f"{flag} is never emitted"
