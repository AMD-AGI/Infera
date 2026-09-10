###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What expert parallelism costs must follow the message, not a constant.

The all-reduce was measured on hardware and the all-to-all was not, and the
asymmetry was expensive. The all-to-all charged ``max(30 us, bandwidth)`` from a
guessed constant, and a decode message never gets large enough for the bandwidth
term to win, so the constant set the price at every batch size. A gpt-oss step
runs 36 MoE layers with a dispatch and a combine each, making that 72 x 30 us =
2.16 ms of fixed cost per step against a measured step of about 3 ms -- while
turning expert parallelism on at batch 1 measures at about 0.25 ms.

Two failures followed, and the second is the serious one. The projection priced
EP as catastrophic at low batch: -63% at TP4 batch 1 against -10% measured. And
because a constant does not grow with the batch while the compute it is compared
against does, the same error made EP look like an increasingly good idea at high
batch, where measurement says it stops helping. That inverted real ranking
decisions, which is the one failure mode a search cannot absorb -- it does not
care much about absolute latency, but it acts on the ordering.

The fix charges the measured bandwidth term only, for the same reason the
all-reduce does: the fitted per-call floor is overwhelmingly the per-kernel
occupancy the decode step already charges for every kernel it launches, so
charging it again counts it twice.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec


def _tpot(ep: int, batch: int, tp: int = 4) -> float:
    return project_spec(
        model="gpt_oss_120B",
        concurrency=batch,
        input_len=1024,
        output_len=1024,
        weight_dtype="mxfp4",
        kv_cache_dtype="bf16",
        tp=tp,
        ep=ep,
    )["tpot_ms"]


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_expert_parallelism_is_not_ruinous_at_batch_one(tp: int):
    """Turning EP on at batch 1 costs a little, and never a lot.

    This was briefly bounded above at 10% on Qwen3-Next-80B and MiniMax-M2
    anchors, which put the net step *faster* with EP on. Those are other models:
    gpt-oss, which is what this projects, settles it the other way. Its EP-on and
    EP-off ladders have since been re-measured at three seeds with no forced
    router skew, so they are the same workload and comparable -- and they put EP
    at +8.1% (TP2), +10.8% (TP4) and +8.8% (TP8) at batch 1, turning slightly
    negative by batch 32.

    The purpose of the bound is unchanged -- the constant all-to-all floor put
    this at +63%, and anything approaching that means the fixed term is back.
    """
    cost = (_tpot(tp, 1, tp=tp) - _tpot(1, 1, tp=tp)) / _tpot(1, 1, tp=tp) * 100.0
    assert 0.0 < cost < 20.0, (
        f"EP costs {cost:.1f}% at tp{tp} batch 1; skew-matched measurement on "
        f"gpt-oss says roughly +8% to +11%"
    )


@pytest.mark.parametrize("world", [2, 4, 8])
def test_the_all_to_all_grows_with_the_message(world: int):
    """Cost is proportional to the bytes that cross a link.

    Asserted on the collective rather than on the step, because the step also
    carries the MLP saving expert parallelism buys, and at high batch that
    saving is the larger term -- a net step delta would be measuring both and
    attributing it to one.

    Under the constant, this function was flat: every message from 6 KB to 3 MB
    cost the same 30 us, which is what made the all-to-all shrink as a share of
    a growing step and turned it into a spurious argument for EP at high batch.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        _measured_intra_node_a2a_us,
    )

    small = _measured_intra_node_a2a_us(64 * 1024, world)
    large = _measured_intra_node_a2a_us(4 * 1024 * 1024, world)
    assert large > small * 8.0, (
        f"world {world}: 64 KB costs {small:.2f} us and 4 MB costs "
        f"{large:.2f} us; cost must follow the message"
    )


@pytest.mark.parametrize("world", [2, 4, 8])
def test_the_all_to_all_charges_no_fixed_floor(world: int):
    """A vanishing message must cost vanishingly little.

    The floor is what a *standalone* collective costs, and most of it is the
    per-kernel occupancy the decode step already charges for every kernel it
    runs. Charging it here as well counted it twice, 72 times per step.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        _measured_intra_node_a2a_us,
    )

    assert _measured_intra_node_a2a_us(1024, world) < 1.0


def test_wider_expert_parallelism_moves_bytes_faster():
    """Measured bandwidth rises with world size: 57 GB/s at 2 ranks, 290 at 8.

    More links carry the exchange. A model that ignored world size would charge
    EP8 the same as EP2 and steer a search away from the wider split for no
    physical reason.
    """
    from infera.projection.core.projection.inference_projection.collectives import (
        _measured_intra_node_a2a_us,
    )

    msg = 4 * 1024 * 1024
    assert (_measured_intra_node_a2a_us(msg, 8)
            < _measured_intra_node_a2a_us(msg, 2))


def test_decode_step_stays_smooth_across_the_batch_axis():
    """No cliff where a message size crosses a threshold.

    Batch is the axis a tuning agent searches, so a step change in cost makes it
    stop at the cliff and report the batch below it as optimal. Asserted on
    shape because a straddling error can flatter aggregate accuracy.
    """
    batches = [8, 16, 32, 64, 128, 256]
    curve = [_tpot(4, b) for b in batches]
    for (b0, t0), (b1, t1) in zip(zip(batches, curve), zip(batches[1:], curve[1:])):
        growth = t1 / t0
        assert growth < 2.2, (
            f"decode jumps {growth:.2f}x from batch {b0} to {b1}; "
            "the batch axis must not contain a cliff"
        )
