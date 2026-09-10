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


def test_disagg_ttft_is_batched_prefill_not_a_fifo_of_singles():
    """A dedicated prefill replica batches its share; it does not FIFO C singles.

    Putting every decode-resident client through ``_closed_loop_wait_ms`` on
    an uncontended single-request prefill made disagg TTFT track C * S and
    saturate 100x high on the GB300 agentic fleets. A batched forward of the
    per-replica share has to stay well below that sweep.
    """
    from .conftest import project_spec

    common = dict(disaggregate=True, prefill_tp=4, tp=8, ep=1,
                  prefill_replicas=1, decode_replicas=1, input_len=1024,
                  output_len=128, prefix_cache_hit_rate=0.0)
    one = project_spec(**common, concurrency=1)
    many = project_spec(**common, concurrency=64)
    assert one["ttft_ms"] > 0 and many["ttft_ms"] > 0
    # FIFO of 64 singles would be ~64x the uncontended service. Batched
    # prefill may grow with batch but not like a sweep.
    assert many["ttft_ms"] < 20.0 * one["ttft_ms"], (
        f"disagg TTFT at C=64 is {many['ttft_ms']:.1f} ms vs "
        f"{one['ttft_ms']:.1f} ms at C=1; a FIFO of singles would be ~64x"
    )
    assert many["ttft_ms"] >= one["ttft_ms"], (
        "a larger in-flight batch cannot make first-token cheaper"
    )


def test_disagg_long_decode_does_not_prefill_every_resident_client():
    """Decode-resident clients are think time, not a prefill batch of size C.

    Pricing TTFT at ``prefill_latency_ms(C / prefill_replicas)`` assumed the
    whole in-flight population was sitting on the prefill GPUs. With a long
    generation that occupancy collapses; first-token has to stay within a
    small factor of the uncontended service, not track C.
    """
    from .conftest import project_spec

    common = dict(disaggregate=True, prefill_tp=4, tp=8, ep=1,
                  prefill_replicas=1, decode_replicas=1, input_len=512,
                  output_len=2048, prefix_cache_hit_rate=0.0)
    one = project_spec(**common, concurrency=1)
    many = project_spec(**common, concurrency=64)
    assert one["ttft_ms"] > 0 and many["ttft_ms"] > 0
    assert many["ttft_ms"] < 8.0 * one["ttft_ms"], (
        f"disagg TTFT at C=64 / OSL=2048 is {many['ttft_ms']:.1f} ms vs "
        f"{one['ttft_ms']:.1f} ms at C=1; decode-resident clients were "
        f"still being priced as a prefill batch"
    )


def test_disagg_more_prefill_replicas_cut_ttft_under_load():
    """Splitting the same load across more prefill workers must reduce TTFT."""
    from .conftest import project_spec

    common = dict(disaggregate=True, prefill_tp=2, tp=8, ep=1,
                  decode_replicas=1, input_len=2048, output_len=128,
                  concurrency=64, prefix_cache_hit_rate=0.0)
    thin = project_spec(**common, prefill_replicas=1)
    wide = project_spec(**common, prefill_replicas=4)
    assert wide["ttft_ms"] < thin["ttft_ms"], (
        f"4 prefill replicas {wide['ttft_ms']:.1f} ms vs 1 replica "
        f"{thin['ttft_ms']:.1f} ms at the same system concurrency"
    )

