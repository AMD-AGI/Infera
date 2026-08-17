###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""A decode step has to get more expensive as the context it reads gets longer.

Hyperloom sweeps context out to 65,536 tokens, and the projection was close to
flat across that whole range for gpt-oss: measured, the step grows 2.26x between
1k and 16k of context at batch 64, where the model grew 1.49x and came out 28.8%
low. The cause was crediting gpt-oss's 128-token sliding window on half its
layers. vLLM only delivers that saving through its hybrid KV-cache manager,
which was not active on any run behind this model -- it reported 2,751,958 KV
tokens for gpt-oss-120b, which is all 36 layers holding full context rather than
18 of them windowed.

So decode now charges the full read. These assert the shape that fixed, against
the measured sweep in bench/hyperloom_validation/context_cost.py, and not on
exact milliseconds -- the point is that the curve climbs, not that any one
number is right.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec

# gpt-oss-120b, TP1, MI355X, batch 64, prefix caching off, three seeds.
# Prompt length -> measured decode step, ms.
MEASURED = {1024: 10.44, 4096: 12.29, 16384: 23.57, 65536: 64.01}

# The windowed model produced 3.05x across this range where the measurement
# shows 6.13x. Anything at or below this is the old flat behaviour returning.
MIN_GROWTH_1K_TO_64K = 4.4


def step(input_len: int, batch: int = 64) -> float:
    return project_spec(
        model="gpt_oss_120B",
        tp=1,
        ep=1,
        concurrency=batch,
        input_len=input_len,
        output_len=256,
        weight_dtype="mxfp4",
    )["decode_step_ms"]


@pytest.fixture(scope="module")
def curve():
    return {il: step(il) for il in MEASURED}


def test_the_step_climbs_with_context(curve):
    lens = sorted(curve)
    for shorter, longer in zip(lens, lens[1:]):
        assert curve[longer] > curve[shorter], (
            f"{longer} tokens of context priced no higher than {shorter}"
        )


def test_the_climb_is_steep_enough_to_match_the_measurement(curve):
    """The window credit made this 3.05x against a measured 6.13x.

    64k is the far end of what Hyperloom searches, and where crediting the
    window left the projection at just over half the real cost.
    """
    growth = curve[65536] / curve[1024]
    assert growth >= MIN_GROWTH_1K_TO_64K, (
        f"decode grew only {growth:.2f}x from 1k to 64k context; the measured "
        f"growth is {MEASURED[65536] / MEASURED[1024]:.2f}x"
    )


@pytest.mark.parametrize("input_len", sorted(MEASURED))
def test_each_point_lands_near_its_measurement(curve, input_len):
    """Loose bound: this is one model against one sweep, not a fit."""
    err = abs(curve[input_len] - MEASURED[input_len]) / MEASURED[input_len]
    assert err < 0.20, (
        f"{input_len} tokens: projected {curve[input_len]:.2f} ms against "
        f"{MEASURED[input_len]:.2f} ms measured ({err * 100:.0f}%)"
    )


def test_a_short_context_is_still_cheap(curve):
    """The fix must not have simply made every decode step more expensive.

    Batch 64 at 1k context is where the ladder was already accurate, so it is
    the regression this change most plausibly causes.
    """
    assert curve[1024] < 1.5 * MEASURED[1024]
