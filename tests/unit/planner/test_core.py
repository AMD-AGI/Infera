###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The scaling decision itself.

Every expected replica count below is worked out in the comment above it from
the flat profiling fixture, so a change in the arithmetic shows up as a failure
with a readable explanation rather than an opaque number mismatch.
"""

from __future__ import annotations

import pytest

from infera.planner.args import PlannerArgs
from infera.planner.core import SlaPlanner
from infera.planner.metrics_source import LoadMetrics
from infera.planner.perf_model import PerfModel


def make_args(**overrides) -> PlannerArgs:
    defaults = {
        "ttft_ms": 500.0,
        "itl_ms": 20.0,
        "adjustment_interval": 100.0,
        "min_endpoint": 1,
        "max_gpu_budget": 1_000,
        "load_predictor": "constant",
        "profile_results": "unused-in-these-tests",
    }
    defaults.update(overrides)
    return PlannerArgs(**defaults)


def make_planner(flat_profile, **overrides) -> SlaPlanner:
    return SlaPlanner(make_args(**overrides), PerfModel(flat_profile))


def metrics(**overrides) -> LoadMetrics:
    """Traffic that the flat fixture predicts exactly, so corrections are 1.0.

    ISL 1000 predicts 100ms TTFT; observing 0.1s makes the prefill correction
    1.0. Concurrency here works out to 1 request per decode replica, where the
    fixture predicts 10ms ITL; observing 0.01s makes the decode correction 1.0.
    """
    defaults = {
        "num_req": 100.0,
        "isl": 1_000.0,
        "osl": 100.0,
        "ttft": 0.1,
        "itl": 0.01,
        "num_prefill": 1,
        "num_decode": 1,
    }
    defaults.update(overrides)
    return LoadMetrics(**defaults)


class TestCorrectionFactors:
    def test_matching_the_profile_gives_unity(self, flat_profile):
        planner = make_planner(flat_profile)
        m = metrics()
        # request_duration = 0.1 + 99 * 0.01 = 1.09s; over a 100s interval with
        # 100 requests on 1 replica that is ~1.09 concurrent requests, which at
        # 1000-token contexts is ~1% KV usage -> clamped to the 10% column, 10ms.
        p, d = planner.correction_factors(m)
        assert p == pytest.approx(1.0)
        assert d == pytest.approx(1.0)

    def test_queueing_shows_up_as_a_prefill_correction_above_one(self, flat_profile):
        planner = make_planner(flat_profile)
        # Same prompt length, but TTFT is three times what an idle engine gives:
        # the extra 200ms is queue time the profiling sweep never saw.
        p, _ = planner.correction_factors(metrics(ttft=0.3))
        assert p == pytest.approx(3.0)

    def test_cache_hits_show_up_as_a_prefill_correction_below_one(self, flat_profile):
        planner = make_planner(flat_profile)
        # Prompts are 1000 tokens but prefix hits mean far less is prefilled, so
        # TTFT beats the profiled figure.
        p, _ = planner.correction_factors(metrics(ttft=0.05))
        assert p == pytest.approx(0.5)

    def test_slow_decode_shows_up_as_a_decode_correction_above_one(self, flat_profile):
        planner = make_planner(flat_profile)
        d = planner.correction_factors(metrics(itl=0.02))[1]
        assert d == pytest.approx(2.0)


class TestPrefillReplicas:
    def test_token_rate_divided_by_profiled_throughput(self, flat_profile):
        planner = make_planner(flat_profile)
        # 100 requests x 1000 tokens over 100s = 1000 prefill tok/s. The fixture
        # sustains 10000 tok/s/gpu on 1 GPU, so one replica covers it.
        planner.observe(metrics())
        assert planner.plan(metrics()).num_prefill == 1

    def test_scales_with_the_token_rate(self, flat_profile):
        planner = make_planner(flat_profile)
        # 2500 requests x 1000 tokens over 100s = 25000 tok/s -> ceil(25000/10000) = 3.
        m = metrics(num_req=2_500.0)
        planner.observe(m)
        assert planner.plan(m).num_prefill == 3

    def test_queueing_does_not_inflate_demand(self, flat_profile):
        planner = make_planner(flat_profile)
        # The prefill correction is capped at 1.0: a TTFT inflated by queueing
        # is what extra replicas fix, so it must not also multiply the demand.
        m = metrics(num_req=2_500.0, ttft=0.5)
        planner.observe(m)
        assert planner.plan(m).prefill_correction == pytest.approx(5.0)
        assert planner.plan(m).num_prefill == 3

    def test_cache_hits_reduce_the_replicas_needed(self, flat_profile):
        planner = make_planner(flat_profile)
        # A 0.2 correction means only a fifth of each prompt is really prefilled:
        # 25000 tok/s x 0.2 = 5000 -> ceil(5000/10000) = 1.
        m = metrics(num_req=2_500.0, ttft=0.02)
        planner.observe(m)
        decision = planner.plan(m)
        assert decision.prefill_correction == pytest.approx(0.2)
        assert decision.num_prefill == 1

    def test_multi_gpu_replicas_need_fewer_of_them(self, flat_profile):
        planner = make_planner(flat_profile, prefill_engine_num_gpu=4)
        # 25000 tok/s over 10000 tok/s/gpu = 2.5 GPUs -> ceil(2.5/4) = 1 replica.
        m = metrics(num_req=2_500.0)
        planner.observe(m)
        assert planner.plan(m).num_prefill == 1


class TestDecodeReplicas:
    def test_token_rate_divided_by_the_throughput_that_fits_the_itl(self, flat_profile):
        planner = make_planner(flat_profile)
        # 100 requests x 100 output tokens over 100s = 100 decode tok/s. The
        # fixture sustains 1000 tok/s/gpu within a 20ms ITL, so one replica does.
        m = metrics()
        planner.observe(m)
        assert planner.plan(m).num_decode == 1

    def test_scales_with_the_output_token_rate(self, flat_profile):
        planner = make_planner(flat_profile)
        # 5000 requests x 100 tokens over 100s = 5000 tok/s -> ceil(5000/1000) = 5.
        m = metrics(num_req=5_000.0)
        planner.observe(m)
        assert planner.plan(m).num_decode == 5

    def test_multi_gpu_replicas_need_fewer_of_them(self, flat_profile):
        planner = make_planner(flat_profile, decode_engine_num_gpu=2)
        # 5000 tok/s over 1000 tok/s/gpu = 5 GPUs -> ceil(5/2) = 3 replicas.
        m = metrics(num_req=5_000.0)
        planner.observe(m)
        assert planner.plan(m).num_decode == 3


class TestDecodeItlTarget:
    """How the decode correction feeds back into the ITL target.

    Exercised through ``_plan_decode`` rather than ``plan``, because the
    correction factor is itself a function of the request rate: raising traffic
    to make the replica arithmetic interesting also moves the concurrency, and
    therefore the very correction under test. Passing the correction in directly
    isolates the behaviour.
    """

    def setup_method(self):
        # Throughput rises with KV utilisation while ITL degrades with it, which
        # is what makes a stricter target cost real capacity: 20ms buys
        # 2000 tok/s/gpu, 10ms only 1000, 5ms only 200.
        from infera.planner.profile_data import parse_profile_data

        profile = parse_profile_data(
            {
                "prefill": {"isl": [1000], "ttft_ms": [100.0], "thpt_per_gpu": [10000.0]},
                "decode": {
                    "kv_usage": [0.1, 0.5, 0.9],
                    "context_length": [1000, 2000],
                    "itl_ms": [[5.0, 10.0, 20.0], [5.0, 10.0, 20.0]],
                    "thpt_per_gpu": [[200.0, 1000.0, 2000.0], [200.0, 1000.0, 2000.0]],
                    "max_kv_tokens": 100_000,
                },
            }
        )
        self.planner = SlaPlanner(make_args(itl_ms=20.0), PerfModel(profile))

    def _replicas(self, correction: float) -> int:
        # 20000 requests x 100 output tokens over 100s = 20000 decode tok/s.
        return self.planner._plan_decode(20_000.0, 1_000.0, 100.0, correction)

    def test_an_on_target_deployment_uses_the_full_sla(self):
        # Correction 1.0 -> the 20ms target stands -> 2000 tok/s/gpu -> 10 replicas.
        assert self._replicas(1.0) == 10

    def test_a_slow_deployment_is_given_a_stricter_target(self):
        on_target = self._replicas(1.0)
        # A correction of 2.0 means the deployment runs at twice the profiled
        # ITL, so the planner aims at 10ms to land on 20ms. That halves the
        # usable per-GPU throughput and doubles the replica count. The search
        # walks a discrete KV grid and takes the last point strictly within
        # budget, so it may land one replica on the conservative side.
        assert self._replicas(2.0) in (2 * on_target, 2 * on_target + 1)
        # Four times slower tightens the target to 5ms, which buys only
        # 200 tok/s/gpu -- a tenth of what 20ms allows.
        assert self._replicas(4.0) == 10 * on_target

    def test_a_fast_deployment_is_allowed_to_run_hotter(self):
        # Beating the profiled ITL loosens the target, which cannot exceed the
        # most loaded profiled point -- so this floors at the same 10 replicas
        # rather than inventing capacity beyond the sweep.
        assert self._replicas(0.5) == 10


class TestSkippedIntervals:
    def test_an_idle_window_leaves_the_deployment_alone(self, flat_profile):
        planner = make_planner(flat_profile)
        assert planner.plan(metrics(num_req=0.0)) is None

    def test_a_window_with_no_tokens_generated_is_not_actionable(self, flat_profile):
        planner = make_planner(flat_profile)
        assert planner.plan(metrics(osl=0.0)) is None

    def test_traffic_without_decode_replicas_cannot_be_calibrated(self, flat_profile):
        # Concurrency is per decode replica; with none registered there is
        # nothing to measure the decode model against.
        planner = make_planner(flat_profile)
        assert planner.plan(metrics(num_decode=0)) is None

    def test_a_window_without_latency_samples_is_not_actionable(self, flat_profile):
        # TTFT and ITL are streaming-only measurements, so a workload of
        # non-streaming requests reports real ISL/OSL and zero latency. Acting
        # on that would read as an infinitely fast fleet.
        planner = make_planner(flat_profile)
        heavy = metrics(num_req=10_000.0, ttft=0.0, itl=0.0)
        planner.observe(heavy)
        assert planner.plan(heavy) is None

    def test_a_missing_itl_alone_is_enough_to_skip(self, flat_profile):
        # Every reply was a single token, so there is no inter-token interval to
        # have measured; the decode correction would come out at zero.
        planner = make_planner(flat_profile)
        assert planner.plan(metrics(itl=0.0)) is None

    def test_a_busy_window_is_never_shrunk_to_the_floor_for_want_of_latency(self, flat_profile):
        # The regression this guards: 10000 requests x 1000 tokens over 100s
        # needs 10 prefill replicas. With the latency averages at zero the
        # prefill correction is 0, which multiplies that demand away to one.
        planner = make_planner(flat_profile)
        measured = metrics(num_req=10_000.0)
        planner.observe(measured)
        assert planner.plan(measured).num_prefill == 10
        assert planner.plan(metrics(num_req=10_000.0, ttft=0.0, itl=0.0)) is None


class TestLimits:
    def test_min_endpoint_keeps_both_pools_alive(self, flat_profile):
        planner = make_planner(flat_profile, min_endpoint=2)
        # Traffic this light asks for one replica each; the floor lifts both.
        m = metrics(num_req=1.0)
        planner.observe(m)
        decision = planner.plan(m)
        assert decision.num_prefill == 2
        assert decision.num_decode == 2

    def test_gpu_budget_caps_the_total(self, flat_profile):
        # Uncapped: 100000 prefill tok/s over 10000 tok/s/gpu = 10 prefill, and
        # 10000 decode tok/s over 1000 tok/s/gpu = 10 decode, so 20 GPUs.
        planner = make_planner(flat_profile)
        m = metrics(num_req=10_000.0, isl=1_000.0, osl=100.0)
        planner.observe(m)
        uncapped = planner.plan(m)
        assert uncapped.num_prefill + uncapped.num_decode == 20
        assert not uncapped.gpu_budget_exceeded

        capped = make_planner(flat_profile, max_gpu_budget=6)
        capped.observe(m)
        decision = capped.plan(m)
        assert decision.gpu_budget_exceeded
        assert decision.num_prefill + decision.num_decode <= 6

    def test_staying_within_budget_is_not_flagged(self, flat_profile):
        planner = make_planner(flat_profile, max_gpu_budget=1_000)
        m = metrics()
        planner.observe(m)
        assert not planner.plan(m).gpu_budget_exceeded

    def test_budget_split_never_overshoots(self, flat_profile):
        # Prefill replicas cost 4 GPUs each, decode 2, against a budget of 10.
        planner = make_planner(
            flat_profile,
            prefill_engine_num_gpu=4,
            decode_engine_num_gpu=2,
            max_gpu_budget=10,
        )
        m = metrics(num_req=20_000.0)
        planner.observe(m)
        decision = planner.plan(m)
        assert decision.num_prefill * 4 + decision.num_decode * 2 <= 10

    def test_min_endpoint_wins_over_the_budget(self, flat_profile):
        # A deliberately undersized budget cannot be honoured without scaling a
        # pool to zero, which would take the deployment down. The floor holds
        # and the overrun is reported instead.
        planner = make_planner(
            flat_profile, min_endpoint=2, max_gpu_budget=2, decode_engine_num_gpu=1
        )
        m = metrics(num_req=10_000.0)
        planner.observe(m)
        decision = planner.plan(m)
        assert decision.num_prefill >= 2
        assert decision.num_decode >= 2
        assert decision.gpu_budget_exceeded


class TestPrediction:
    def test_the_forecast_drives_the_decision_not_the_last_window(self, flat_profile):
        planner = make_planner(flat_profile, load_predictor="ewma")
        # A quiet history followed by one spike: an EWMA forecast lands between
        # the two, so the planner neither ignores the spike nor chases it.
        for _ in range(5):
            planner.observe(metrics(num_req=100.0))
        spike = metrics(num_req=10_000.0)
        planner.observe(spike)
        decision = planner.plan(spike)
        assert 100.0 < decision.predicted_num_req < 10_000.0

    def test_a_constant_predictor_repeats_the_last_window(self, flat_profile):
        planner = make_planner(flat_profile, load_predictor="constant")
        m = metrics(num_req=2_500.0)
        planner.observe(m)
        assert planner.plan(m).predicted_num_req == pytest.approx(2_500.0)


class TestDecisionShape:
    def test_records_the_fleet_it_observed(self, flat_profile):
        planner = make_planner(flat_profile)
        m = metrics(num_prefill=3, num_decode=7)
        planner.observe(m)
        decision = planner.plan(m)
        assert decision.observed_prefill == 3
        assert decision.observed_decode == 7
        assert decision.changes_anything

    def test_a_no_op_decision_is_recognised(self, flat_profile):
        planner = make_planner(flat_profile)
        m = metrics()
        planner.observe(m)
        decision = planner.plan(m)
        assert decision.num_prefill == 1 and decision.num_decode == 1
        assert not decision.changes_anything

    def test_summary_names_both_pools(self, flat_profile):
        planner = make_planner(flat_profile)
        m = metrics()
        planner.observe(m)
        summary = planner.plan(m).summary()
        assert "prefill" in summary and "decode" in summary
