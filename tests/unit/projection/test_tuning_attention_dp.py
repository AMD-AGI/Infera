###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The serving search has to be able to express attention data parallelism.

Every knob the tuner sweeps is one it can recommend, and every knob it does not
is one it will silently rule out. Attention DP was in the projector and the
memory model but not in the search, so a tuner asked for the best way to serve
an MLA model could only ever answer with configurations that store the same
latent cache once per rank -- which is not how any of these models is deployed.
"""

from __future__ import annotations

from infera.projection.agents.tuning_agent.inference_tuning import (
    InferenceTrialConfig,
    derive_inference_legality,
    validate_inference,
)


class _Arch:
    """The fields the legality derivation reads off a workload."""

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


def _legality():
    return derive_inference_legality(_Arch(), _Cluster())


def _cfg(**kw) -> InferenceTrialConfig:
    base = dict(tp=8, ep=1, batch_size=16, weight_dtype="bf16", kv_cache_dtype="bf16")
    base.update(kw)
    return InferenceTrialConfig(**base)


def test_the_axis_is_offered_at_every_degree_that_divides_the_tp_group():
    leg = _legality()
    assert leg.attention_dp == [1, 2, 4, 8], (
        f"attention DP degrees {leg.attention_dp} for a TP group of {max(leg.tp)}"
    )


def test_the_agent_is_told_the_axis_exists():
    """It plans over the prompt dict; an axis missing from it is unsearchable."""
    assert "attention_dp" in _legality().to_prompt_dict()


def test_a_degree_that_divides_the_group_is_legal():
    leg = _legality()
    ok, why = validate_inference(_cfg(attention_dp=8), _Arch(), _Cluster(), leg)
    assert ok, why


def test_a_degree_that_does_not_divide_the_group_is_rejected_not_rounded():
    """The projector raises on this; the tuner must not spend a trial finding out."""
    leg = _legality()
    ok, why = validate_inference(_cfg(tp=4, attention_dp=8), _Arch(), _Cluster(), leg)
    assert not ok and "divide" in why


def test_off_is_still_the_default():
    """Every trial that does not ask for the axis must project as it always did."""
    assert InferenceTrialConfig().attention_dp == 1
    leg = _legality()
    ok, why = validate_inference(_cfg(), _Arch(), _Cluster(), leg)
    assert ok, why


def test_the_seed_plan_spends_trials_on_the_axis():
    """Seeded rather than left to the agent: it decides MLA serving outright.

    The plan truncates, so an axis seeded late is an axis not tried at all.
    """
    from infera.projection.agents.tuning_agent.inference_tuning import (
        build_inference_seed_plan,
    )

    class _Opt:
        input_len = 8192
        output_len = 1024
        max_concurrency = 64
        hbm_capacity_gb = 288.0
        memory_safety_margin = 0.05
        slo = {}
        objective = "decode_tps_per_gpu"

    plan = build_inference_seed_plan(_Arch(), _Cluster(), _Opt(), max_candidates=16)
    dp_trials = [c for c in plan.candidates if c.attention_dp > 1]
    assert dp_trials, "no seed trial turns attention DP on"
    # And not only at batch 1, where the freed capacity has nothing to hold.
    assert any(c.batch_size >= 16 for c in dp_trials)


def test_a_draft_model_that_is_never_wrong_and_never_costs_anything_is_rejected():
    """Speculation has to pay for itself or the optimizer will take a free lunch.

    Acceptance and draft cost are free parameters. Handed a draft that is always
    right and runs for nothing, the search took it and reported 354 tok/s/GPU on
    a configuration no draft model can implement -- 2.5x the honest number.
    """
    leg = _legality()
    perfect = _cfg(
        speculative_num_tokens=4, speculative_acceptance_rate=1.0, speculative_draft_cost_factor=0.0
    )
    ok, why = validate_inference(perfect, _Arch(), _Cluster(), leg)
    assert not ok and "acceptance_rate" in why

    free = _cfg(
        speculative_num_tokens=4, speculative_acceptance_rate=0.7, speculative_draft_cost_factor=0.0
    )
    ok, why = validate_inference(free, _Arch(), _Cluster(), leg)
    assert not ok and "draft cost" in why


def test_speculation_that_pays_for_itself_is_still_allowed():
    """The axis stays searchable; only the unphysical corner is closed."""
    leg = _legality()
    cfg = _cfg(
        speculative_num_tokens=4, speculative_acceptance_rate=0.7, speculative_draft_cost_factor=0.2
    )
    ok, why = validate_inference(cfg, _Arch(), _Cluster(), leg)
    assert ok, why


def test_accuracy_has_to_be_bought():
    """Acceptance and cost describe one draft model, so they move together.

    Barring only the perfect-and-free corner left the class of problem intact:
    acceptance 0.95 at a cost of 0.10 was legal, which is the same free lunch
    one step less brazen.
    """
    from infera.projection.agents.tuning_agent.inference_tuning import min_draft_cost

    leg = _legality()
    cheap = _cfg(
        speculative_num_tokens=4,
        speculative_acceptance_rate=0.95,
        speculative_draft_cost_factor=0.10,
    )
    ok, why = validate_inference(cheap, _Arch(), _Cluster(), leg)
    assert not ok and "too cheap" in why

    # The cost floor rises with acceptance and leaves no reason to speculate
    # once the draft costs a whole target step.
    assert min_draft_cost(0.7) < min_draft_cost(0.9) < min_draft_cost(0.95)
    assert min_draft_cost(0.96) > 1.0


def test_the_projection_command_carries_the_axis():
    """Legal in the search but undelivered to the projector would score every
    DP trial as if the axis were off, which is worse than not searching it."""
    from infera.projection.agents.tuning_agent import evaluator as ev

    cmd = " ".join(str(x) for x in _fake_cmd(ev, attention_dp=8))
    assert "--attention-dp-size 8" in cmd
    assert "--attention-dp-size" not in " ".join(str(x) for x in _fake_cmd(ev))


def _fake_cmd(ev, **kw):
    """Build the inference command for one trial without touching the filesystem."""
    import types

    cfg = _cfg(**kw)
    agent_cfg = types.SimpleNamespace(
        target_cluster=types.SimpleNamespace(
            gpu_arch="mi355x", gpu_clock_mhz=None, gpus_per_node=8, num_nodes=1
        ),
        optimization=types.SimpleNamespace(hbm_capacity_gb=288.0),
    )
    from pathlib import Path

    return ev._build_inference_cmd(Path("w.yaml"), cfg, agent_cfg, Path("."))
