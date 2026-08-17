###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Windowing of the server's cumulative /metrics into interval averages.

The exposition text here is hand-built rather than captured from a live server
so the expected averages are arithmetic the reader can follow.
"""

from __future__ import annotations

import httpx
import pytest

from infera.planner.metrics_source import (
    LoadMetrics,
    MetricsSource,
    Snapshot,
    parse_metrics_text,
)


def exposition(
    *,
    requests: float,
    ttft_sum: float,
    itl_sum: float,
    isl_sum: float,
    osl_sum: float,
    sla_count: float | None = None,
    prefill: int = 2,
    decode: int = 4,
    failed_requests: float = 0.0,
) -> str:
    """Render an Infera server's /metrics for the samples the planner reads.

    ``sla_count`` defaults to ``requests``: every successful request observes
    each SLA histogram once.
    """
    count = requests if sla_count is None else sla_count
    return f"""\
# HELP infera_request_duration_seconds End-to-end latency.
# TYPE infera_request_duration_seconds histogram
infera_request_duration_seconds_bucket{{router="disagg",outcome="ok",le="+Inf"}} {requests}
infera_request_duration_seconds_sum{{router="disagg",outcome="ok"}} 12.0
infera_request_duration_seconds_count{{router="disagg",outcome="ok"}} {requests}
infera_request_duration_seconds_sum{{router="disagg",outcome="5xx"}} 3.0
infera_request_duration_seconds_count{{router="disagg",outcome="5xx"}} {failed_requests}
# HELP infera_time_to_first_token_seconds TTFT.
# TYPE infera_time_to_first_token_seconds histogram
infera_time_to_first_token_seconds_sum{{router="disagg"}} {ttft_sum}
infera_time_to_first_token_seconds_count{{router="disagg"}} {count}
# HELP infera_inter_token_latency_seconds ITL.
# TYPE infera_inter_token_latency_seconds histogram
infera_inter_token_latency_seconds_sum{{router="disagg"}} {itl_sum}
infera_inter_token_latency_seconds_count{{router="disagg"}} {count}
# HELP infera_input_sequence_tokens ISL.
# TYPE infera_input_sequence_tokens histogram
infera_input_sequence_tokens_sum{{router="disagg"}} {isl_sum}
infera_input_sequence_tokens_count{{router="disagg"}} {count}
# HELP infera_output_sequence_tokens OSL.
# TYPE infera_output_sequence_tokens histogram
infera_output_sequence_tokens_sum{{router="disagg"}} {osl_sum}
infera_output_sequence_tokens_count{{router="disagg"}} {count}
# HELP infera_active_workers Active workers.
# TYPE infera_active_workers gauge
infera_active_workers{{disagg_mode="prefill"}} {prefill}
infera_active_workers{{disagg_mode="decode"}} {decode}
infera_active_workers{{disagg_mode="mixed"}} 0
"""


def source_over(pages: list[list[str]]) -> MetricsSource:
    """A MetricsSource whose scrapes return ``pages[n]``, one body per URL.

    Each ``collect()`` consumes one entry, so a test can hand the source a
    scripted sequence of scrape rounds.
    """
    rounds = list(pages)

    def handler(request: httpx.Request) -> httpx.Response:
        if not rounds:
            raise AssertionError("scraped more rounds than the test scripted")
        bodies = rounds[0]
        index = int(request.url.params.get("replica", "0"))
        body = bodies[index]
        if index == len(bodies) - 1:
            rounds.pop(0)
        return httpx.Response(200, text=body)

    urls = [f"http://server/metrics?replica={i}" for i in range(len(pages[0]))]
    source = MetricsSource(urls)
    source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return source


class TestParseMetricsText:
    def test_reads_the_sla_histograms_and_worker_gauges(self):
        snapshot = Snapshot()
        parse_metrics_text(
            exposition(requests=10, ttft_sum=2.0, itl_sum=0.5, isl_sum=10_000, osl_sum=1_000),
            snapshot,
        )
        assert snapshot.totals_for("infera_time_to_first_token_seconds") == (2.0, 10.0)
        assert snapshot.totals_for("infera_input_sequence_tokens") == (10_000.0, 10.0)
        assert snapshot.active_prefill == 2
        assert snapshot.active_decode == 4

    def test_failed_requests_are_excluded_from_the_request_count(self):
        # The SLA histograms only observe successes, so pairing them with a
        # count that included 5xx would understate every average.
        snapshot = Snapshot()
        parse_metrics_text(
            exposition(
                requests=10,
                ttft_sum=2.0,
                itl_sum=0.5,
                isl_sum=10_000,
                osl_sum=1_000,
                failed_requests=7,
            ),
            snapshot,
        )
        assert snapshot.totals_for("infera_request_duration_seconds")[1] == 10.0

    def test_histograms_sum_across_replicas_but_worker_gauges_do_not(self):
        snapshot = Snapshot()
        for _ in range(3):
            parse_metrics_text(
                exposition(requests=10, ttft_sum=2.0, itl_sum=0.5, isl_sum=10_000, osl_sum=1_000),
                snapshot,
            )
        assert snapshot.totals_for("infera_time_to_first_token_seconds") == (6.0, 30.0)
        # Every server watches the same fleet; summing would triple it.
        assert snapshot.active_decode == 4


class TestCollect:
    async def test_first_scrape_only_establishes_a_baseline(self):
        page = exposition(requests=10, ttft_sum=2.0, itl_sum=0.5, isl_sum=10_000, osl_sum=1_000)
        source = source_over([[page], [page]])
        assert await source.collect() is None
        await source.aclose()

    async def test_averages_are_the_window_delta(self):
        first = exposition(requests=10, ttft_sum=2.0, itl_sum=0.5, isl_sum=10_000, osl_sum=1_000)
        # 20 more requests: TTFT 0.3s, ITL 0.02s, ISL 2000, OSL 100 each.
        second = exposition(
            requests=30,
            ttft_sum=2.0 + 20 * 0.3,
            itl_sum=0.5 + 20 * 0.02,
            isl_sum=10_000 + 20 * 2_000,
            osl_sum=1_000 + 20 * 100,
        )
        source = source_over([[first], [second]])
        assert await source.collect() is None

        metrics = await source.collect()
        assert metrics is not None
        assert metrics.num_req == pytest.approx(20.0)
        assert metrics.ttft == pytest.approx(0.3)
        assert metrics.itl == pytest.approx(0.02)
        assert metrics.isl == pytest.approx(2_000.0)
        assert metrics.osl == pytest.approx(100.0)
        assert metrics.num_prefill == 2
        assert metrics.num_decode == 4
        await source.aclose()

    async def test_averages_span_the_whole_fleet(self):
        # Two replicas, each serving 10 requests in the window: one at 0.1s
        # TTFT and one at 0.5s, so the fleet average is 0.3s.
        base = exposition(requests=0, ttft_sum=0.0, itl_sum=0.0, isl_sum=0, osl_sum=0)
        fast = exposition(
            requests=10, ttft_sum=10 * 0.1, itl_sum=10 * 0.01, isl_sum=10 * 1_000, osl_sum=10 * 50
        )
        slow = exposition(
            requests=10, ttft_sum=10 * 0.5, itl_sum=10 * 0.03, isl_sum=10 * 3_000, osl_sum=10 * 150
        )
        source = source_over([[base, base], [fast, slow]])
        assert await source.collect() is None

        metrics = await source.collect()
        assert metrics is not None
        assert metrics.num_req == pytest.approx(20.0)
        assert metrics.ttft == pytest.approx(0.3)
        assert metrics.isl == pytest.approx(2_000.0)
        await source.aclose()

    async def test_counter_reset_invalidates_the_window(self):
        # A server restart zeroes its counters. Differencing across that would
        # produce a negative delta, so the interval is skipped entirely.
        high = exposition(requests=100, ttft_sum=30.0, itl_sum=5.0, isl_sum=100_000, osl_sum=10_000)
        restarted = exposition(requests=1, ttft_sum=0.3, itl_sum=0.02, isl_sum=2_000, osl_sum=100)
        source = source_over([[high], [restarted], [restarted]])
        assert await source.collect() is None
        assert await source.collect() is None
        await source.aclose()

    async def test_window_recovers_after_a_reset(self):
        high = exposition(requests=100, ttft_sum=30.0, itl_sum=5.0, isl_sum=100_000, osl_sum=10_000)
        restarted = exposition(requests=1, ttft_sum=0.3, itl_sum=0.02, isl_sum=2_000, osl_sum=100)
        after = exposition(requests=3, ttft_sum=0.9, itl_sum=0.06, isl_sum=6_000, osl_sum=300)
        source = source_over([[high], [restarted], [after]])
        assert await source.collect() is None  # baseline
        assert await source.collect() is None  # reset detected
        metrics = await source.collect()
        assert metrics is not None
        assert metrics.num_req == pytest.approx(2.0)
        assert metrics.ttft == pytest.approx(0.3)
        await source.aclose()

    async def test_idle_window_reports_zero_rather_than_dividing_by_zero(self):
        page = exposition(requests=10, ttft_sum=2.0, itl_sum=0.5, isl_sum=10_000, osl_sum=1_000)
        source = source_over([[page], [page]])
        assert await source.collect() is None
        metrics = await source.collect()
        assert metrics is not None
        assert metrics.num_req == 0.0
        assert metrics.ttft == 0.0
        assert not metrics.has_traffic
        await source.aclose()

    async def test_unreachable_fleet_yields_nothing(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        source = MetricsSource(["http://server/metrics"])
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert await source.collect() is None
        await source.aclose()

    async def test_one_dead_replica_does_not_blind_the_planner(self):
        page = exposition(requests=10, ttft_sum=3.0, itl_sum=0.2, isl_sum=10_000, osl_sum=1_000)

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("replica") == "1":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, text=page)

        source = MetricsSource(
            ["http://server/metrics?replica=0", "http://server/metrics?replica=1"]
        )
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        snapshot = await source.scrape()
        assert snapshot.scraped == 1
        assert snapshot.totals_for("infera_time_to_first_token_seconds") == (3.0, 10.0)
        await source.aclose()

    def test_requires_at_least_one_url(self):
        with pytest.raises(ValueError, match="at least one"):
            MetricsSource([])


class TestLoadMetrics:
    def test_request_duration_is_rebuilt_from_ttft_and_itl(self):
        # 0.3s to the first token, then 99 more at 20ms.
        metrics = LoadMetrics(
            num_req=10, isl=1_000, osl=100, ttft=0.3, itl=0.02, num_prefill=1, num_decode=1
        )
        assert metrics.request_duration == pytest.approx(0.3 + 99 * 0.02)

    def test_single_token_reply_has_no_decode_time(self):
        metrics = LoadMetrics(
            num_req=10, isl=1_000, osl=1, ttft=0.3, itl=0.0, num_prefill=1, num_decode=1
        )
        assert metrics.request_duration == pytest.approx(0.3)

    @pytest.mark.parametrize(
        ("num_req", "isl", "osl"),
        [(0, 1_000, 100), (10, 0, 100), (10, 1_000, 0)],
    )
    def test_a_window_missing_any_dimension_is_not_actionable(self, num_req, isl, osl):
        metrics = LoadMetrics(
            num_req=num_req,
            isl=isl,
            osl=osl,
            ttft=0.3,
            itl=0.02,
            num_prefill=1,
            num_decode=1,
        )
        assert not metrics.has_traffic
