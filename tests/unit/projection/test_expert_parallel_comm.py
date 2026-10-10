###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The expert all-to-all is sized by the expert group, not by what surrounds it.

``ep`` ranks perform this collective and the group size is handed to the
collective model directly. The other parallelism arguments only describe the
domain it crosses, so widening tensor parallelism around a fixed expert group
must not change what the dispatch costs -- the same ranks are exchanging the
same bytes either way.

The projector used to pass ``tp // ep`` as that domain argument, which the
collective multiplies by the group size to decide whether the transfer stays on
a node. At TP16/EP8 that made the test read ``16 <= 8`` and priced a dispatch
between eight GPUs of one node at pod bandwidth. Nothing measured caught it:
every deployment in the campaign runs ``ep`` equal to ``tp`` or no expert
parallelism at all, so ``tp // ep`` was always 1 and the fault only appeared in
projected geometries where the two widths differ.

Asserted on shape, not on values. Crossing a node is allowed to cost more; being
told a group crosses one when it does not is the fault under test.
"""

from __future__ import annotations

import pytest

# One node is 8 GPUs, so an EP8 group fits exactly and EP16 genuinely spans two.
NODE_SIZE = 8
TOKENS = 4096


def _a2a_ms(tp, ep, tokens=TOKENS):
    """Dispatch + combine for one MoE layer at this parallelism, in ms."""
    from infera.projection.core.projection.inference_projection.collectives import (
        InferenceCollectiveModel,
    )
    from infera.projection.core.projection.training_config import (
        InferenceCollectiveConfig,
    )

    class _Model:
        hidden_size = 6144
        moe_router_topk = 8

    class _Parallel:
        tensor_model_parallel_size = tp
        pipeline_model_parallel_size = 1
        expert_model_parallel_size = ep
        context_model_parallel_size = 1
        attention_data_parallel_size = 1

    model = InferenceCollectiveModel(_Model(), _Parallel(), InferenceCollectiveConfig(enabled=True))
    return model.ep_a2a_ms(1, tokens)


def test_attention_dp_sizes_dispatch_to_tokens_on_rank():
    """A rank under DP attention only all-to-alls its own sequences.

    Replica batch 512 with DP=8 is the same payload as batch 64 with DP=1, not
    an 8x larger collective on every rank.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        InferenceCollectiveModel,
    )
    from infera.projection.core.projection.training_config import (
        InferenceCollectiveConfig,
    )

    class _Model:
        hidden_size = 7168
        moe_router_topk = 8

    def _ms(dp: int, batch: int) -> float:
        class _Parallel:
            tensor_model_parallel_size = 8
            pipeline_model_parallel_size = 1
            expert_model_parallel_size = 8
            context_model_parallel_size = 1
            attention_data_parallel_size = dp

        model = InferenceCollectiveModel(
            _Model(), _Parallel(), InferenceCollectiveConfig(enabled=True)
        )
        return model.ep_a2a_ms(batch, 1)

    replica = _ms(8, 512)
    per_rank = _ms(1, 64)
    unsplit = _ms(1, 512)
    assert replica == pytest.approx(per_rank, rel=0.05)
    assert replica < 0.5 * unsplit


@pytest.mark.parametrize("tp", [8, 16, 32])
def test_tensor_width_does_not_change_an_on_node_expert_dispatch(tp):
    """An EP8 group is eight GPUs of one node whatever TP is wrapped around it."""
    base = _a2a_ms(NODE_SIZE, NODE_SIZE)
    got = _a2a_ms(tp, NODE_SIZE)
    assert got == pytest.approx(base, rel=0.05), (
        f"TP{tp}/EP{NODE_SIZE} prices the dispatch at {got:.2f} ms against "
        f"{base:.2f} ms at TP{NODE_SIZE}/EP{NODE_SIZE}; the same eight ranks "
        "exchange the same bytes, so the tensor width around them is being "
        "charged to the expert group"
    )


def test_a_group_inside_one_node_is_not_priced_as_leaving_it():
    """The on-node group must stay far below the one that genuinely crosses.

    Pinned as an ordering rather than a ratio: the point is that EP8 is charged
    like a node-local transfer, not that any millisecond figure is right.
    """
    inside = _a2a_ms(16, NODE_SIZE)
    outside = _a2a_ms(16, NODE_SIZE * 2)
    assert inside < outside, (
        f"an EP{NODE_SIZE} dispatch ({inside:.2f} ms) is not cheaper than an "
        f"EP{NODE_SIZE * 2} one ({outside:.2f} ms), so the node boundary is not "
        "being detected at all"
    )


def test_widening_experts_within_a_node_stays_sane():
    """Cost must not double with every expert-parallel rung inside one node.

    The regression this guards ran ~2x per doubling of EP because the domain
    argument fell as EP rose, so each rung was priced one domain further out.
    Over the MoE layers of a large model that turned prefill into ~90%
    communication and made every wide-expert prefill pool look unusable.
    """
    curve = [_a2a_ms(NODE_SIZE, ep) for ep in (2, 4, 8)]
    assert curve == sorted(curve), f"expert dispatch is not monotone in EP: {curve}"
    assert curve[-1] < curve[0] * 3.0, (
        f"EP8 costs {curve[-1] / curve[0]:.1f}x EP2 inside one node "
        f"({curve[0]:.2f} -> {curve[-1]:.2f} ms); an all-to-all moves "
        "(n-1)/n of its buffer, so the curve flattens rather than doubling"
    )
