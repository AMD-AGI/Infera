###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A warmup should ask for as few GPUs as it can and still be useful.

The rule is deliberately simple: measure a config on its own footprint when that
fits in four GPUs, and on four when it does not. Anything larger is projected
rather than measured, which is what keeps a warmup to one short run instead of a
ladder over every parallelism a search might ask about.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.inference_projection.benchmark_vllm import (
    WARMUP_GPU_CAP,
    warmup_gpu_count,
)


@pytest.mark.parametrize(
    "tp, ep, expected",
    [
        (1, 1, 1),    # a single-GPU config is measured on one GPU
        (2, 1, 2),
        (4, 1, 4),    # exactly the cap
        # A TP=2 EP=2 target occupies two GPUs in vLLM, where EP follows TP, so
        # the warmup cannot be wider than the thing it is measuring.
        (2, 2, 2),
        (8, 1, 4),    # beyond the cap: measure on four, project the rest
        (8, 8, 4),
        (16, 1, 4),   # multi-node targets never enlarge the warmup
        (4, 4, 4),
    ],
)
def test_a_warmup_never_asks_for_more_than_half_a_node(tp, ep, expected):
    assert warmup_gpu_count(tp, ep) == expected
    assert warmup_gpu_count(tp, ep) <= WARMUP_GPU_CAP


def test_the_warmup_size_stays_a_parallelism_the_model_can_run():
    """TP has to divide the target, so a degree the cap does not divide steps
    down rather than asking for a split that cannot be built."""
    assert warmup_gpu_count(3, 1) == 3   # fits under the cap on its own
    assert warmup_gpu_count(6, 1) == 2   # 4 does not divide 6
    assert warmup_gpu_count(12, 1) == 4


def test_a_larger_target_never_costs_a_larger_warmup():
    """The whole point is that warmup cost stops growing with the search space."""
    sizes = [warmup_gpu_count(tp, 1) for tp in (1, 2, 4, 8, 16, 32, 64)]
    assert max(sizes) == WARMUP_GPU_CAP
    assert sizes == sorted(sizes), "warmup size should not oscillate with target"
