###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Decode cost must be smooth in batch size.

A message-size threshold in the collective model used to switch in a
training-scale overhead partway up the batch axis, so DeepSeek-R1 decode jumped
2.2x between batch 144 and 160. Batch is the axis the tuning agent searches, and
a step that size dominates every real effect on it: the agent stops at the cliff
and reports the batch just below it as optimal. Aggregate error against measured
can even look *better* with such a step in place, because the projection
straddles the truth instead of sitting under it, which is why this is asserted
on shape rather than on accuracy.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec

# Straddles the 2 MiB all-reduce message size where the old threshold sat.
BATCHES = [96, 112, 128, 144, 160, 176, 192, 224, 256]


@pytest.fixture(scope="module")
def decode_curve():
    return [
        project_spec(
            model="deepseek_v3",
            concurrency=b,
            input_len=1024,
            output_len=1024,
            weight_dtype="fp8",
            kv_cache_dtype="fp8",
        )["tpot_ms"]
        for b in BATCHES
    ]


def test_decode_cost_rises_with_batch(decode_curve):
    assert decode_curve == sorted(decode_curve)


def test_no_step_change_between_adjacent_batches(decode_curve):
    """Neighbouring batch sizes are ~15% apart, so no step should exceed that much."""
    for lo, hi, before, after in zip(BATCHES, BATCHES[1:], decode_curve, decode_curve[1:]):
        assert after / before < 1.35, (
            f"decode cost jumps {after / before:.2f}x from batch {lo} to {hi} "
            f"({before:.2f} -> {after:.2f} ms)"
        )


def test_curve_is_gentler_than_the_batch_axis(decode_curve):
    """Decode is weight-bound, so 2.7x the batch must cost well under 2.7x."""
    assert decode_curve[-1] / decode_curve[0] < 2.0
