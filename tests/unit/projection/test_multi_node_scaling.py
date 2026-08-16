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

    Cross-node all-reduce is genuinely more expensive than intra-node, so this
    is not asserting the boundary is free. It is asserting the boundary costs
    something like a communication hop rather than something like an order of
    magnitude: 460 us of RCCL setup and NIC warmup per layer per step put 22 ms
    of fixed cost into a 53 ms token on a 48-layer model.
    """
    inside = wide_curve[NODE_SIZE]
    outside = wide_curve[NODE_SIZE * 2]
    assert outside < inside * 1.5, (
        f"TP{NODE_SIZE * 2} costs {outside / inside:.1f}x TP{NODE_SIZE} "
        f"({inside:.2f} -> {outside:.2f} ms); the node boundary is being charged "
        "a training-scale fixed overhead per layer"
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
