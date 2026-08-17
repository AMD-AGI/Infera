###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Load predictors."""

from __future__ import annotations

import pytest

from infera.planner.predictor import (
    PREDICTORS,
    ConstantPredictor,
    EwmaPredictor,
    build_predictor,
)


class TestConstantPredictor:
    def test_predicts_zero_before_any_observation(self):
        assert ConstantPredictor().predict_next() == 0.0

    def test_repeats_the_last_observation(self):
        p = ConstantPredictor()
        for value in (10.0, 20.0, 35.0):
            p.add_data_point(value)
        assert p.predict_next() == 35.0


class TestEwmaPredictor:
    def test_weights_the_newest_observation_by_alpha(self):
        p = EwmaPredictor(alpha=0.5)
        p.add_data_point(10.0)
        p.add_data_point(20.0)
        # 0.5 * 20 + 0.5 * 10
        assert p.predict_next() == pytest.approx(15.0)

    def test_smooths_a_single_spike(self):
        p = EwmaPredictor(alpha=0.5)
        for value in (10.0, 10.0, 10.0, 100.0):
            p.add_data_point(value)
        # The spike moves the estimate but does not become the estimate.
        assert 10.0 < p.predict_next() < 100.0

    def test_alpha_of_one_degenerates_to_constant(self):
        p = EwmaPredictor(alpha=1.0)
        for value in (10.0, 20.0, 35.0):
            p.add_data_point(value)
        assert p.predict_next() == pytest.approx(35.0)

    @pytest.mark.parametrize("alpha", [0.0, -0.5, 1.5])
    def test_rejects_an_alpha_outside_zero_to_one(self, alpha):
        with pytest.raises(ValueError, match="alpha"):
            EwmaPredictor(alpha=alpha)


class TestHistoryHandling:
    def test_leading_zeros_are_dropped(self):
        # Before traffic arrives the metrics window is empty. Keeping those
        # zeros would hold the forecast down through the first real interval.
        p = ConstantPredictor()
        p.add_data_point(0.0)
        p.add_data_point(0.0)
        assert p.history == []
        p.add_data_point(50.0)
        assert p.history == [50.0]

    def test_zeros_after_traffic_are_kept(self):
        # An idle interval mid-stream is real signal: load genuinely dropped.
        p = ConstantPredictor()
        p.add_data_point(50.0)
        p.add_data_point(0.0)
        assert p.history == [50.0, 0.0]
        assert p.predict_next() == 0.0

    def test_history_is_bounded_by_the_window(self):
        p = ConstantPredictor(window_size=3)
        for value in range(1, 11):
            p.add_data_point(float(value))
        assert p.history == [8.0, 9.0, 10.0]


class TestRegistry:
    @pytest.mark.parametrize("name", sorted(PREDICTORS))
    def test_every_registered_name_builds(self, name):
        predictor = build_predictor(name, window_size=5)
        predictor.add_data_point(7.0)
        assert predictor.predict_next() == pytest.approx(7.0)

    def test_unknown_name_lists_the_valid_ones(self):
        with pytest.raises(ValueError, match="constant"):
            build_predictor("arima")
