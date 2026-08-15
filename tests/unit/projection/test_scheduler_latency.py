###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Engine knobs that trade first-token latency for host efficiency.

Production serving configs raise ``--stream-interval`` and batch decode steps
between scheduler polls; neither costs throughput, which is why they get turned
up, and neither is visible to a FLOPs model. On Infera's own 1P1D DeepSeek-R1
sweep they are most of the measured TTFT, so a projector that ignores them
reports ~50 ms against a measured ~2.5 s and would rank a latency-SLO
configuration on noise.
"""

from __future__ import annotations

import pytest

from .conftest import project_spec

BASE = dict(input_len=1024, output_len=256, concurrency=32)


def test_defaults_are_inert():
    """Both knobs off must reproduce the plain projection exactly."""
    plain = project_spec(**BASE)
    explicit = project_spec(**BASE, stream_interval=1, decode_admission_steps=0)
    assert explicit["ttft_ms"] == pytest.approx(plain["ttft_ms"])
    assert explicit["tpot_ms"] == pytest.approx(plain["tpot_ms"])


def test_stream_interval_moves_latency_from_tpot_into_ttft():
    """The client waits the same total time; only the split changes."""
    plain = project_spec(**BASE)
    buffered = project_spec(**BASE, stream_interval=30)
    assert buffered["ttft_ms"] > plain["ttft_ms"]
    assert buffered["tpot_ms"] < plain["tpot_ms"]
    assert buffered["e2el_ms"] == pytest.approx(plain["e2el_ms"], rel=1e-6)


def test_stream_interval_costs_about_one_flush():
    plain = project_spec(**BASE)
    buffered = project_spec(**BASE, stream_interval=30)
    added = buffered["ttft_ms"] - plain["ttft_ms"]
    assert added == pytest.approx(29 * plain["tpot_ms"], rel=0.05)


def test_admission_granularity_is_real_added_time():
    """Unlike buffering, waiting for the scheduler to look extends the request."""
    plain = project_spec(**BASE)
    polled = project_spec(**BASE, decode_admission_steps=80)
    assert polled["ttft_ms"] > plain["ttft_ms"]
    assert polled["e2el_ms"] > plain["e2el_ms"]


def test_admission_wait_averages_half_the_window():
    plain = project_spec(**BASE)
    polled = project_spec(**BASE, decode_admission_steps=80)
    added = polled["ttft_ms"] - plain["ttft_ms"]
    assert added == pytest.approx(40 * plain["tpot_ms"], rel=0.15)


def test_neither_knob_changes_throughput():
    plain = project_spec(**BASE)
    tuned = project_spec(**BASE, stream_interval=30, decode_admission_steps=80)
    assert tuned["decode_tps"] == pytest.approx(plain["decode_tps"], rel=1e-6)
