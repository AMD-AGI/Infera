###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Interpolation over the profiled performance grids."""

from __future__ import annotations

import pytest

from infera.planner.perf_model import DecodePerfModel, PerfModel, PrefillPerfModel
from infera.planner.profile_data import parse_profile_data


class TestPrefillPerfModel:
    def setup_method(self):
        self.model = PrefillPerfModel(
            parse_profile_data(
                {
                    "prefill": {
                        "isl": [1000, 2000, 4000],
                        "ttft_ms": [100.0, 200.0, 500.0],
                        "thpt_per_gpu": [9000.0, 10000.0, 11000.0],
                    },
                    "decode": {
                        "kv_usage": [0.5],
                        "context_length": [1000],
                        "itl_ms": [[10.0]],
                        "thpt_per_gpu": [[1000.0]],
                        "max_kv_tokens": 100_000,
                    },
                }
            ).prefill
        )

    def test_hits_the_profiled_points_exactly(self):
        assert self.model.interpolate_ttft(2000) == pytest.approx(200.0)
        assert self.model.interpolate_thpt_per_gpu(4000) == pytest.approx(11000.0)

    def test_interpolates_linearly_between_points(self):
        assert self.model.interpolate_ttft(1500) == pytest.approx(150.0)
        assert self.model.interpolate_thpt_per_gpu(3000) == pytest.approx(10500.0)

    def test_clamps_instead_of_extrapolating(self):
        # Extrapolating a saturating throughput curve would invent capacity the
        # hardware does not have, so both ends pin to the profiled edge.
        assert self.model.interpolate_ttft(1) == pytest.approx(100.0)
        assert self.model.interpolate_ttft(1_000_000) == pytest.approx(500.0)
        assert self.model.interpolate_thpt_per_gpu(0) == pytest.approx(9000.0)
        assert self.model.interpolate_thpt_per_gpu(99_999) == pytest.approx(11000.0)


def _decode_model(**overrides) -> DecodePerfModel:
    decode = {
        "kv_usage": [0.2, 0.6, 1.0],
        "context_length": [1000, 3000],
        # ITL degrades as KV fills, and more steeply at long context.
        "itl_ms": [
            [10.0, 20.0, 30.0],
            [20.0, 40.0, 60.0],
        ],
        "thpt_per_gpu": [
            [500.0, 1000.0, 1200.0],
            [400.0, 800.0, 900.0],
        ],
        "max_kv_tokens": 100_000,
    }
    decode.update(overrides)
    data = parse_profile_data(
        {
            "prefill": {"isl": [1000], "ttft_ms": [100.0], "thpt_per_gpu": [10000.0]},
            "decode": decode,
        }
    )
    return DecodePerfModel(data.decode)


class TestDecodePerfModelInterpolation:
    def setup_method(self):
        self.model = _decode_model()

    def test_kv_usage_is_concurrency_times_context_over_capacity(self):
        # 20 requests x 1000 tokens / 100000 capacity = 20%.
        assert self.model.kv_usage_for(20, 1000) == pytest.approx(0.2)

    def test_kv_usage_clamps_to_the_profiled_range(self):
        assert self.model.kv_usage_for(0, 1000) == pytest.approx(0.2)
        assert self.model.kv_usage_for(10_000, 1000) == pytest.approx(1.0)

    def test_hits_grid_corners_exactly(self):
        # 60 x 1000 / 100000 = 60% kv usage at the first context-length row.
        assert self.model.interpolate_itl(60, 1000) == pytest.approx(20.0)
        assert self.model.interpolate_thpt_per_gpu(60, 1000) == pytest.approx(1000.0)

    def test_bilinear_across_both_axes(self):
        # context_length 2000 is halfway between the two profiled rows, and
        # 40 x 2000 / 100000 = 80% kv usage is halfway between columns 0.6 and
        # 1.0. ITL there: rows give 25 and 50, midpoint 37.5.
        assert self.model.interpolate_itl(40, 2000) == pytest.approx(37.5)

    def test_context_length_clamps_to_the_profiled_range(self):
        # Below the first profiled row, values pin to that row rather than
        # extrapolating down to an impossibly fast ITL.
        assert self.model.interpolate_itl(20, 10) == pytest.approx(
            self.model.interpolate_itl(20, 1000)
        )


class TestFindBestThroughputPerGpu:
    def setup_method(self):
        self.model = _decode_model()

    def test_picks_the_most_loaded_point_within_budget(self):
        # At context 1000 the ITL curve runs 10 -> 20 -> 30 as KV fills. A 20ms
        # budget admits up to 60% usage, where throughput is 1000 tok/s/gpu.
        thpt, itl, kv = self.model.find_best_throughput_per_gpu(20.0, 1000)
        assert itl == pytest.approx(20.0, abs=0.5)
        assert kv == pytest.approx(0.6, abs=0.02)
        assert thpt == pytest.approx(1000.0, abs=20.0)

    def test_a_looser_budget_buys_more_throughput(self):
        tight, _, _ = self.model.find_best_throughput_per_gpu(20.0, 1000)
        loose, _, _ = self.model.find_best_throughput_per_gpu(30.0, 1000)
        assert loose > tight

    def test_unreachable_budget_returns_the_lightest_point(self):
        # Nothing on the curve meets 1ms; the caller detects this from the
        # returned ITL exceeding what it asked for.
        thpt, itl, kv = self.model.find_best_throughput_per_gpu(1.0, 1000)
        assert itl > 1.0
        assert kv == pytest.approx(0.2)
        assert thpt == pytest.approx(500.0)

    def test_scans_downward_so_a_non_monotonic_dip_cannot_win(self):
        # A dip at high load would let a bisection settle there and overstate
        # capacity. The downward scan must still stop at the 0.6 column, whose
        # throughput is lower than the dip's.
        model = _decode_model(
            itl_ms=[
                [10.0, 20.0, 15.0],
                [10.0, 20.0, 15.0],
            ],
            thpt_per_gpu=[
                [500.0, 1000.0, 3000.0],
                [500.0, 1000.0, 3000.0],
            ],
        )
        thpt, itl, kv = model.find_best_throughput_per_gpu(16.0, 1000)
        assert kv == pytest.approx(1.0)
        assert itl == pytest.approx(15.0)
        assert thpt == pytest.approx(3000.0)
        # Whereas a budget below the dip lands back on the low-load side.
        _, itl_tight, kv_tight = model.find_best_throughput_per_gpu(11.0, 1000)
        assert kv_tight < 0.4
        assert itl_tight <= 11.0


class TestPerfModel:
    def test_carries_the_profiled_gpu_counts(self, flat_profile):
        model = PerfModel(flat_profile)
        assert model.prefill_engine_num_gpu == 1
        assert model.decode_engine_num_gpu == 1
        assert model.prefill.interpolate_ttft(1000) == pytest.approx(100.0)
        assert model.decode.interpolate_itl(10, 1000) == pytest.approx(10.0)
