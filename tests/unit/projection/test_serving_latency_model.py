###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The TTFT model must hold in both load regimes, not just the saturated one.

TTFT used to be priced at a single uncontended prompt, which understates it by
the queue depth, and was then briefly priced at a full FIFO sweep, which
overstates it whenever generation is long enough to keep the prefill stage idle.
Real Hyperloom workloads sit on both sides of that line -- gpt-oss at OSL 128 is
prefill-heavy, MiniMax at OSL 1500 is not -- so the queue has to interpolate.
"""

from __future__ import annotations

import pytest

from infera.projection.core.projection.inference_projection.performance import (
    InferencePerformanceProjector as Proj,
)

wait = Proj._closed_loop_wait_ms


def test_single_client_never_queues():
    assert wait(10.0, 500.0, 1) == pytest.approx(10.0)


def test_idle_stage_does_not_queue():
    """Generation so long that prompts never overlap: TTFT is the bare prefill."""
    assert wait(10.0, 1_000_000.0, 256) == pytest.approx(10.0, rel=1e-2)


def test_saturated_stage_approaches_the_fifo_sweep():
    """No think time: the C-th client waits behind all C-1 others."""
    assert wait(10.0, 0.0, 64) == pytest.approx(640.0, rel=1e-6)


def test_wait_is_monotonic_in_clients():
    waits = [wait(10.0, 500.0, c) for c in (1, 8, 32, 128, 512)]
    assert waits == sorted(waits)


def test_wait_is_bounded_by_the_sweep():
    """Queueing can never cost more than prefilling everyone ahead of you."""
    for clients in (2, 16, 256):
        assert wait(7.5, 300.0, clients) <= 7.5 * clients + 1e-9


def test_think_time_relieves_the_queue():
    loaded = wait(10.0, 50.0, 32)
    relieved = wait(10.0, 5000.0, 32)
    assert relieved < loaded
