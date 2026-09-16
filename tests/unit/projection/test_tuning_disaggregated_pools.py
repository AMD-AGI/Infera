###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The serving search has to be able to propose the disaggregated shape these
models are actually served in.

A knob the tuner cannot express is a configuration it will silently rule out,
and disaggregation was ruled out three times over. The one split the plan
derived asked for ``max(TP)`` GPUs on the prefill pool and then a decode pool on
top, which on a single node needs nine GPUs out of eight: illegal, and dropped
by ``add()`` without a word, so a one-node search never priced the topology at
all. On two nodes it became legal but sat at position 297 of 307 against a
default seed budget of 12, so it was still never scored. And no candidate
anywhere carried both ``disaggregate`` and an attention-DP degree, because the
trial config had no per-pool field and ``mk()`` defaults the global one to 1.

That last one is what made the gap expensive rather than merely incomplete.
GLM-5.2 and DeepSeek-V4 both cache a single latent that every head reads, which
tensor parallelism *replicates* rather than shards -- so a TP8 decode pool at
attention-DP 1 stores the same cache eight times, and the decode pool's
concurrency ceiling, the whole reason to disaggregate, was understated
eightfold. The search was choosing between colocated configurations that got
data-parallel attention and a disaggregated one that could not.
"""

from __future__ import annotations

import pytest

from infera.projection.agents.tuning_agent.inference_tuning import (
    InferenceTrialConfig,
    build_inference_seed_plan,
    derive_inference_legality,
    disagg_splits,
    validate_inference,
)


class _Arch:
    """The fields the legality derivation and seed plan read off a workload."""

    workload_path = "unused.yaml"
    model_name = "glm5"
    num_attention_heads = 64
    hidden_size = 6144
    num_layers = 78
    is_moe = True
    num_experts = 256
    moe_router_topk = 8
    index_topk = 2048


class _Cluster:
    gpu_arch = "mi355x"
    gpu_clock_mhz = None

    def __init__(self, nodes: int = 1, gpus: int = 8):
        self.num_nodes = nodes
        self.gpus_per_node = gpus


class _Opt:
    """The serving profile the plan sweeps around."""

    hbm_capacity_gb = 288.0
    memory_safety_margin = 0.10
    objective = "max_throughput"
    slo: dict = {}
    inference = {"input_len": 4096, "output_len": 256, "max_concurrency": None}


def _legality(cluster=None):
    return derive_inference_legality(_Arch(), cluster or _Cluster())


def _cfg(**kw) -> InferenceTrialConfig:
    base = dict(tp=4, ep=1, batch_size=16, weight_dtype="bf16", kv_cache_dtype="bf16")
    base.update(kw)
    return InferenceTrialConfig(**base)


def _plan(cluster=None, budget=12):
    cluster = cluster or _Cluster()
    return build_inference_seed_plan(_Arch(), cluster, _Opt(), max_candidates=budget)


def _is_dpa(c) -> bool:
    return bool(
        getattr(c, "attention_dp", 1) > 1
        or getattr(c, "prefill_attention_dp", None)
        or getattr(c, "decode_attention_dp", None)
    )


# --------------------------------------------------------------------------
# The pools are expressible at all
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    [
        "prefill_attention_dp",
        "decode_attention_dp",
        "prefill_replicas",
        "prefill_ep",
        "decode_ep",
    ],
)
def test_each_pool_can_state_its_own_shape(field):
    """A field the trial config lacks is a configuration the search cannot reach.

    The projector has taken per-pool attention-DP, EP and replica counts all
    along; the tuner simply had no way to ask for them, so the LLM stage could
    not propose one either.
    """
    assert hasattr(InferenceTrialConfig(), field)


def test_a_pool_attention_dp_is_checked_against_its_own_tp():
    """Each degree splits its own pool's tensor-parallel group, not the trial's.

    Checked against ``cfg.tp`` instead, a legal TP4 prefill pool beside a TP8
    decode pool gets rejected, and a degree describing no rank layout in either
    pool gets accepted.
    """
    leg, cluster = _legality(), _Cluster()
    ok, why = validate_inference(
        _cfg(
            tp=2,
            disaggregate=True,
            prefill_tp=4,
            decode_tp=2,
            decode_replicas=2,
            prefill_attention_dp=4,
            decode_attention_dp=2,
        ),
        _Arch(),
        cluster,
        leg,
    )
    assert ok, f"prefill dp4|tp4 with decode dp2|tp2 is a real layout: {why}"

    bad, why = validate_inference(
        _cfg(tp=2, disaggregate=True, prefill_tp=2, decode_tp=2, prefill_attention_dp=4),
        _Arch(),
        cluster,
        leg,
    )
    assert not bad and "prefill_attention_dp" in why, (
        "a degree wider than its own pool's TP describes no rank layout"
    )


def test_both_pools_replica_counts_are_priced_against_the_cluster():
    """``prefill_replicas`` was accepted into the arithmetic as if it were 1.

    Only the decode pool was multiplied by its replica count, so two prefill
    replicas were charged as one and a split that overcommits the cluster
    validated clean.
    """
    ok, why = validate_inference(
        _cfg(tp=4, disaggregate=True, prefill_tp=4, prefill_replicas=2, decode_tp=4),
        _Arch(),
        _Cluster(),
        _legality(),
    )
    assert not ok and "GPUs" in why, f"4x2 prefill + 4 decode is 12 GPUs on 8: {why}"


# --------------------------------------------------------------------------
# The splits actually fit the cluster
# --------------------------------------------------------------------------


@pytest.mark.parametrize("world", [8, 16, 32])
def test_every_proposed_split_fits_and_spends_the_whole_cluster(world):
    """The bug, stated as the arithmetic that caused it.

    ``prefill_tp = max(TP)`` is the whole node, so the decode pool had nowhere
    to go. Leftover GPUs are equally wrong in a topology whose entire argument
    is spending them where they pay.
    """
    splits = disagg_splits(world, [1, 2, 4, 8])
    assert splits, f"no legal split offered for a {world}-GPU cluster"
    for p_tp, d_tp, reps in splits:
        assert p_tp + d_tp * reps == world, (
            f"prefill {p_tp} + decode {d_tp}x{reps} != {world} GPUs"
        )


def test_a_split_is_never_proposed_that_leaves_decode_homeless():
    """The single-node case that silently removed the topology from the search."""
    for p_tp, _, reps in disagg_splits(8, [1, 2, 4, 8]):
        assert p_tp < 8, "prefill cannot take the whole cluster"
        assert reps >= 1


def test_the_two_pool_widths_are_swept_independently():
    """Asymmetry is the topology's argument, so it has to be on the table.

    Taking the first decode width that fit each prefill width returned only the
    diagonal -- (4,4,1) and (2,2,3) on one node -- which prices the two pools as
    a single knob. The shape these deployments actually run is a prefill pool
    wide enough to hold TTFT down beside several narrower decode replicas, and
    it was never proposed.
    """
    shapes = disagg_splits(8, [1, 2, 4, 8], max_splits=6)
    assert any(p != d for p, d, _ in shapes), f"only symmetric pools offered: {shapes}"
    assert (4, 2, 2) in shapes, shapes


def test_the_widest_prefill_pool_is_still_offered_first():
    """Ordering is a priority order, because the seed budget truncates it."""
    shapes = disagg_splits(16, [1, 2, 4, 8], max_splits=4)
    assert shapes[0] == (8, 8, 1), shapes
    prefills = [p for p, _, _ in shapes]
    assert prefills == sorted(prefills, reverse=True), shapes


def test_asymmetric_shapes_still_spend_the_whole_cluster():
    """The constraint that made the diagonal safe must survive widening it."""
    for world in (8, 16, 32):
        for p_tp, d_tp, reps in disagg_splits(world, [1, 2, 4, 8], max_splits=8):
            assert p_tp + d_tp * reps == world, (world, p_tp, d_tp, reps)


@pytest.mark.parametrize("cluster", [_Cluster(1, 8), _Cluster(2, 8)], ids=["1node", "2node"])
def test_every_disaggregated_candidate_the_plan_emits_is_legal(cluster):
    """Illegal candidates are dropped by ``add()`` without a warning, so an
    illegal proposal does not fail the search -- it removes the topology from it
    and reports a colocated winner as if it had competed."""
    leg = derive_inference_legality(_Arch(), cluster)
    plan = _plan(cluster, budget=64)
    disagg = [c for c in plan.candidates if c.disaggregate]
    assert disagg, "the topology must be on the table"
    for c in disagg:
        ok, why = validate_inference(c, _Arch(), cluster, leg)
        assert ok, f"emitted an illegal split: {why}"


# --------------------------------------------------------------------------
# It survives the seed budget, and it carries attention-DP
# --------------------------------------------------------------------------


def test_the_topology_is_scored_at_the_default_seed_budget():
    """Present in the plan but never reached is the same as absent.

    Disaggregation is emitted late because it depends on widths settled earlier,
    and the attention-DP sweep alone produces hundreds of candidates ahead of
    it. Against the default budget of 12 it was cut every time.
    """
    from infera.projection.agents.tuning_agent.cli import _parse_args

    default_budget = _parse_args(
        ["--workload", "w.yaml", "--target-cluster", "c.yaml"]
    ).seed_budget
    plan = _plan(budget=default_budget)

    disagg = [c for c in plan.candidates if c.disaggregate]
    assert disagg, (
        f"no disaggregated candidate within the default budget of {default_budget}"
    )


def test_the_reserved_slots_do_not_displace_the_head_of_the_plan():
    """Reserving is not promoting: the priority order earned by everything that
    differs from the baseline by one knob has to survive intact."""
    budget = 12
    plan = _plan(budget=budget)
    unreserved = _plan(budget=200)  # large enough that nothing is reserved

    head = [c.signature() for c in plan.candidates if not c.disaggregate]
    assert head == [c.signature() for c in unreserved.candidates[: len(head)]], (
        "the non-disaggregated head of the plan changed order"
    )


def test_the_disaggregated_candidates_carry_attention_dp():
    """The expensive half of the gap.

    For a latent-cache model, attention-DP 1 is the decode pool's worst case --
    tensor parallelism replicates the cache -- so scoring the topology only
    there understates the one number it exists to improve.
    """
    plan = _plan(budget=12)
    disagg = [c for c in plan.candidates if c.disaggregate]
    assert any(_is_dpa(c) for c in disagg), (
        "every disaggregated candidate is at attention-DP 1, which is the "
        "layout these models are never served in"
    )


def test_the_topology_is_also_scored_without_attention_dp():
    """The control. Without it the layout's contribution is not readable: a
    disaggregated winner could be winning on the split or on the layout, and
    they are separate decisions."""
    plan = _plan(budget=12)
    disagg = [c for c in plan.candidates if c.disaggregate]
    assert any(not _is_dpa(c) for c in disagg)


def test_the_budget_is_spent_on_different_splits_rather_than_one():
    """Variants of a single pool shape answer a narrower question than shapes do.

    Emitted grouped by split, the reserved slots all went to one shape and no
    second shape was ever priced -- so the search could say whether *that* split
    beat colocated, but not which split to run.
    """
    plan = _plan(_Cluster(2, 8), budget=12)
    shapes = {
        (c.prefill_tp, c.decode_tp, c.decode_replicas)
        for c in plan.candidates
        if c.disaggregate
    }
    assert len(shapes) > 1, f"only one pool shape priced: {shapes}"


# --------------------------------------------------------------------------
# The flags reach the projector
# --------------------------------------------------------------------------


def test_the_per_pool_flags_are_emitted_and_the_projector_accepts_them():
    """A field that never reaches the command line is still unreachable.

    Asserted against the projector's own parser rather than a hardcoded list, so
    a rename on either side fails here instead of silently dropping the knob.
    """
    pytest.importorskip("yaml")
    from infera.projection.agents.tuning_agent.evaluator import _build_inference_cmd
    from infera.projection.cli import build_parser

    class _AgentCfg:
        target_cluster = _Cluster()
        optimization = _Opt()

    cmd = _build_inference_cmd(
        __import__("pathlib").Path("wl.yaml"),
        _cfg(
            tp=2,
            disaggregate=True,
            prefill_tp=4,
            decode_tp=2,
            decode_replicas=2,
            prefill_attention_dp=4,
            decode_attention_dp=2,
            prefill_ep=4,
            decode_ep=2,
        ),
        _AgentCfg(),
        __import__("pathlib").Path("."),
        pool_benchmarks={"prefill": "p.json", "decode": "d.json"},
    )

    for flag in (
        "--prefill-attention-dp",
        "--decode-attention-dp",
        "--prefill-ep",
        "--decode-ep",
        "--prefill-benchmark",
        "--decode-benchmark",
    ):
        assert flag in cmd, f"{flag} never reaches the projector"

    known = {
        s for a in build_parser()._subparsers._group_actions[0].choices["inference"]._actions
        for s in a.option_strings
    }
    for tok in cmd:
        if tok.startswith("--"):
            assert tok in known, f"{tok} is not a flag the projector accepts"
