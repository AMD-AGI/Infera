###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The observe-decide-apply cycle, with the metrics source and connector faked."""

from __future__ import annotations

import pytest

from infera.planner.core import SlaPlanner
from infera.planner.decision import ScalingDecision
from infera.planner.metrics_source import LoadMetrics
from infera.planner.perf_model import PerfModel

from .test_core import make_args


class FakeMetricsSource:
    """Replays a scripted sequence of windows, then reports nothing further."""

    def __init__(self, windows: list[LoadMetrics | None]) -> None:
        self._windows = list(windows)
        self.collect_calls = 0

    async def collect(self) -> LoadMetrics | None:
        self.collect_calls += 1
        return self._windows.pop(0) if self._windows else None

    async def aclose(self) -> None:
        return


class FakeConnector:
    def __init__(self) -> None:
        self.applied: list[ScalingDecision] = []

    async def apply(self, decision: ScalingDecision) -> None:
        self.applied.append(decision)

    async def aclose(self) -> None:
        return


def window(**overrides) -> LoadMetrics:
    defaults = {
        "num_req": 5_000.0,
        "isl": 1_000.0,
        "osl": 100.0,
        "ttft": 0.1,
        "itl": 0.01,
        "num_prefill": 1,
        "num_decode": 1,
    }
    defaults.update(overrides)
    return LoadMetrics(**defaults)


def build(flat_profile, windows, **arg_overrides):
    source = FakeMetricsSource(windows)
    connector = FakeConnector()
    planner = SlaPlanner(
        make_args(**arg_overrides),
        PerfModel(flat_profile),
        metrics_source=source,
        connector=connector,
    )
    return planner, source, connector


class TestTick:
    async def test_a_decision_reaches_the_connector(self, flat_profile):
        planner, _, connector = build(flat_profile, [window()])
        decision = await planner.tick()
        assert decision is not None
        assert connector.applied == [decision]

    async def test_an_unchanged_decision_is_not_applied(self, flat_profile):
        # Patching the deployment with the counts it already has is pure churn:
        # the operator would reconcile a no-op and the log would fill with it.
        planner, _, connector = build(
            flat_profile, [window(num_req=100.0, num_prefill=1, num_decode=1)]
        )
        decision = await planner.tick()
        assert decision is not None
        assert not decision.changes_anything
        assert connector.applied == []

    async def test_no_metrics_means_no_decision(self, flat_profile):
        planner, _, connector = build(flat_profile, [None])
        assert await planner.tick() is None
        assert connector.applied == []

    async def test_an_idle_window_is_observed_but_not_acted_on(self, flat_profile):
        planner, source, connector = build(flat_profile, [window(num_req=0.0)])
        assert await planner.tick() is None
        assert source.collect_calls == 1
        assert connector.applied == []

    async def test_history_accumulates_across_ticks(self, flat_profile):
        # Each tick must feed the predictors, otherwise a smoothing forecaster
        # would never see more than one point.
        planner, _, _ = build(
            flat_profile,
            [window(num_req=1_000.0), window(num_req=2_000.0), window(num_req=3_000.0)],
            load_predictor="ewma",
        )
        for _ in range(3):
            await planner.tick()
        assert planner._num_req_predictor.history == [1_000.0, 2_000.0, 3_000.0]

    async def test_gpu_counts_come_from_the_profile_unless_overridden(self, flat_profile):
        planner, _, _ = build(flat_profile, [window()])
        assert planner.prefill_num_gpu == 1

        override, _, _ = build(flat_profile, [window()], prefill_engine_num_gpu=8)
        assert override.prefill_num_gpu == 8

    async def test_scaling_up_is_driven_by_the_observed_fleet(self, flat_profile):
        # 5000 requests x 100 output tokens over 100s needs 5 decode replicas;
        # the fleet reports 1, so the decision must move it.
        planner, _, connector = build(flat_profile, [window(num_prefill=1, num_decode=1)])
        decision = await planner.tick()
        assert decision.num_decode == 5
        assert decision.observed_decode == 1
        assert connector.applied[0].num_decode == 5

    async def test_scaling_down_when_traffic_drains(self, flat_profile):
        planner, _, connector = build(
            flat_profile, [window(num_req=100.0, num_prefill=4, num_decode=9)]
        )
        decision = await planner.tick()
        assert decision.num_prefill == 1
        assert decision.num_decode == 1
        assert connector.applied[0].num_decode == 1


class TestRunLoopResilience:
    async def test_a_failing_interval_does_not_stop_the_loop(self, flat_profile, monkeypatch):
        # One bad interval must not take the planner down: the next scrape
        # re-establishes the window.
        planner, _, _ = build(flat_profile, [window(), window()])
        calls = {"n": 0}

        async def flaky() -> None:
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("transient")

        monkeypatch.setattr(planner, "tick", flaky)

        async def stop_after_two(_seconds):
            if calls["n"] >= 2:
                raise KeyboardInterrupt

        monkeypatch.setattr("infera.planner.core.asyncio.sleep", stop_after_two)
        with pytest.raises(KeyboardInterrupt):
            await planner.run()
        assert calls["n"] == 2
