###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Going wider than one node has to stay priceable, and priced sanely.

Everything measured in this campaign is intra-node, so multi-node was never
scored and two faults sat in it. Above TP16 the projector raised
ZeroDivisionError, and a search cannot choose a configuration it cannot price at
all. Below that it priced TP16 6.5x slower than TP8 for one model, which is
worse than a crash: the search believes it and rules out multi-node silently.

Both came from constants calibrated on training-shaped collectives -- hundreds
of megabytes, issued rarely -- being charged to a decode step that sends a couple
of megabytes per layer, thousands of times a second. Asserted on shape rather
than on values, since there is no inter-node measurement to assert values
against; the point is that the curve is continuous and monotone, not that any
particular millisecond is right.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec

# One node is 8 GPUs, so TP16 and beyond cross the boundary.
NODE_SIZE = 8
WIDE = [4, 8, 16, 32, 64]


@pytest.fixture(scope="module")
def wide_curve():
    return {
        tp: project_spec(
            model="qwen3_30B_A3B",
            tp=tp,
            ep=tp,
            concurrency=128,
            input_len=1024,
            output_len=1024,
            weight_dtype="bf16",
        )["decode_step_ms"]
        for tp in WIDE
    }


def test_a_multi_node_config_can_be_priced_at_all(wide_curve):
    """Every rung returns a number.

    The contention derate was written as 1 - c*(n-1), which crosses zero and
    took the effective bandwidth with it; and node_size // tp floors to zero
    once TP is wider than a node, which became a zero NIC count. Either one
    raised ZeroDivisionError rather than a slow projection.
    """
    for tp, step in wide_curve.items():
        assert step and step > 0, f"TP{tp} did not project"


def test_crossing_the_node_boundary_is_a_cost_not_a_cliff(wide_curve):
    """The step out of the node should be visible, and only visible.

    Cross-node all-to-all is genuinely more expensive than intra-node, so this
    is not asserting the boundary is free. It is asserting the boundary costs
    something like a communication hop rather than something like an order of
    magnitude: 460 us of RCCL setup and NIC warmup per layer per step put 22 ms
    of fixed cost into a 53 ms token on a 48-layer model.

    The bound is what the hardware ratio supports rather than a round number.
    At EP16 over 8-GPU nodes a rank has 15 peers, 8 of them across a NIC that
    carries roughly a fifth of what the on-node fabric does, and the two halves
    of the exchange run concurrently -- so the collective slows by something
    near the bandwidth ratio on the half that leaves the node, not by it on the
    whole message. That lands the step around 1.6x, and 2x is the point past
    which no blend of those two bandwidths explains the answer any more.
    """
    inside = wide_curve[NODE_SIZE]
    outside = wide_curve[NODE_SIZE * 2]
    assert outside < inside * 2.0, (
        f"TP{NODE_SIZE * 2} costs {outside / inside:.1f}x TP{NODE_SIZE} "
        f"({inside:.2f} -> {outside:.2f} ms); the node boundary is being charged "
        "a training-scale fixed overhead per layer"
    )


def test_the_cross_node_price_is_bandwidth_and_not_a_fixed_cost():
    """Past the node boundary, a bigger message costs proportionally more.

    This is the assertion the ratio above was really standing in for, and it
    holds the model to the shape of the fault rather than to a threshold. A
    per-collective fixed cost -- RCCL setup, NIC warmup, a launch chain charged
    once per layer per step -- does not grow with the message, so if one is
    present the per-byte price falls away as the message grows. Bandwidth is
    flat in per-byte terms. Sweeping the decode dispatch over a 64x range of
    batch separates them without needing to know what the right millisecond is.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        InferenceCollectiveConfig,
        InferenceCollectiveModel,
    )

    class _MC:
        hidden_size = 2048
        num_moe_experts = 128
        moe_router_topk = 8
        num_layers = 48
        use_turbo_deepep = False
        turbo_sync_free_moe_stage = 0

    class _MP:
        tensor_model_parallel_size = 16
        expert_model_parallel_size = 16
        pipeline_model_parallel_size = 1
        context_parallel_size = 1

    model = InferenceCollectiveModel(_MC(), _MP(), InferenceCollectiveConfig(enabled=True))
    per_mb = []
    for batch in (32, 128, 512, 2048):
        msg_mb = batch * _MC.hidden_size * _MC.moe_router_topk * 2 / 2**20
        per_mb.append(model.ep_a2a_ms(batch, 1) / msg_mb)

    # The smallest message here is where a floor would still be allowed to show;
    # compare across the range that is unambiguously bandwidth-bound.
    biggest, smallest = per_mb[-1], per_mb[1]
    assert biggest == pytest.approx(smallest, rel=0.25), (
        f"per-MB cross-node all-to-all price is not flat across batch: {per_mb}; "
        "a fixed per-collective cost is being charged on every step"
    )


def test_going_wider_never_looks_free(wide_curve):
    """Past the node boundary, more nodes cost more.

    A saturating contention derate must not accidentally make a wider config
    cheaper than a narrower one, which would send the search the other way.
    """
    crossed = [wide_curve[tp] for tp in WIDE if tp > NODE_SIZE]
    assert crossed == sorted(crossed), f"decode step is not monotone past one node: {crossed}"


def test_bandwidth_derate_never_reaches_zero():
    """The derate has to survive an arbitrary node count.

    Asserted directly because the crash it caused only appears at widths the
    projection fixtures above would be slow to cover.
    """
    from infera.projection.core.projection.module_profilers.collective_model import (
        remote_contention_factor,
    )

    class _Args:
        a2a_remote_contention = 0.04

    args = _Args()
    factors = [remote_contention_factor(args, n) for n in (1, 2, 4, 26, 64, 1024)]
    assert factors[0] == pytest.approx(1.0), "a single node has nothing to contend with"
    assert all(f > 0 for f in factors), "a link never carries nothing"
    assert factors == sorted(factors, reverse=True), "more nodes must not derate less"
    # Still agrees with the linear form over the two- and four-node range that
    # form was calibrated on, to within about a point of bandwidth -- far inside
    # what the calibration itself resolves, and the reason this is a safe swap
    # rather than a re-fit.
    for nodes in (2, 4):
        assert remote_contention_factor(args, nodes) == pytest.approx(
            1 - 0.04 * (nodes - 1), abs=0.015
        )
