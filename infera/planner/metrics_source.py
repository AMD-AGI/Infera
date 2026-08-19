###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Turn the server fleet's ``/metrics`` into per-interval workload averages.

Rather than depend on a Prometheus deployment, the planner scrapes the Infera
servers directly and does the windowing itself. That keeps a single-host Docker
setup viable and removes PromQL from the critical path; the cost is that the
planner holds the previous scrape in memory, so a planner restart forfeits one
interval.

The histograms Infera exposes are cumulative, so an interval average is
``(sum_now - sum_before) / (count_now - count_before)``. Scrapes from every
server replica are summed before differencing, which is what makes the average
fleet-wide rather than per-replica.

A server restart resets its counters. Detecting that per-series is unreliable
once replicas are summed -- one replica restarting only dents the total -- so
any negative delta invalidates the whole window and the planner skips that
interval rather than acting on a nonsense average.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

# Successful requests only: the SLA histograms are recorded on the ok path, so
# pairing them with a count that included 5xx would understate the averages.
_REQUESTS = "infera_request_duration_seconds"
_TTFT = "infera_time_to_first_token_seconds"
_ITL = "infera_inter_token_latency_seconds"
_ISL = "infera_input_sequence_tokens"
_OSL = "infera_output_sequence_tokens"
_ACTIVE_WORKERS = "infera_active_workers"

_HISTOGRAMS = (_REQUESTS, _TTFT, _ITL, _ISL, _OSL)


@dataclass
class Snapshot:
    """Fleet-summed cumulative values from one scrape round."""

    # metric name -> (sum, count)
    totals: dict[str, tuple[float, float]] = field(default_factory=dict)
    active_prefill: int = 0
    active_decode: int = 0
    active_mixed: int = 0
    scraped: int = 0

    def totals_for(self, metric: str) -> tuple[float, float]:
        return self.totals.get(metric, (0.0, 0.0))


@dataclass(frozen=True)
class LoadMetrics:
    """Workload characteristics observed over one adjustment interval.

    ``ttft`` and ``itl`` are in seconds, matching the exposed histograms;
    :mod:`infera.planner.core` converts where the profiling data uses
    milliseconds.
    """

    num_req: float
    isl: float
    osl: float
    ttft: float
    itl: float
    num_prefill: int
    num_decode: int
    num_mixed: int = 0

    @property
    def has_traffic(self) -> bool:
        """Whether the interval carries enough signal to scale on.

        A window with no completed request, or one that produced no tokens,
        says nothing about whether the SLA is being met.
        """
        return self.num_req > 0 and self.isl > 0 and self.osl > 0

    @property
    def has_latency(self) -> bool:
        """Whether the window recorded the latencies the corrections need.

        Both histograms are streaming-only -- a non-streaming reply arrives in
        one piece, so it has no observable first-token boundary -- and ITL
        additionally needs a reply of at least two tokens. A window that
        recorded neither leaves both averages at zero, which reads as an
        infinitely fast deployment rather than an unmeasured one.
        """
        return self.ttft > 0 and self.itl > 0

    @property
    def request_duration(self) -> float:
        """Mean request latency in seconds, reconstructed from TTFT and ITL.

        Used to convert a request rate into an in-flight concurrency. Derived
        rather than read from ``infera_request_duration_seconds`` because that
        histogram is observed when ``dispatch`` returns, which for a streaming
        reply is at the first token rather than the last -- it would understate
        long generations badly.
        """
        return self.ttft + self.itl * max(0.0, self.osl - 1.0)


def parse_metrics_text(text: str, snapshot: Snapshot) -> None:
    """Accumulate one server's exposition text into ``snapshot``.

    Histogram sums and counts add across replicas. ``infera_active_workers`` is
    not summed: every server watches the same fleet, so the replicas report the
    same number and the widest view wins -- summing would multiply the fleet by
    the replica count.
    """
    wanted_sums = {f"{name}_sum": name for name in _HISTOGRAMS}
    wanted_counts = {f"{name}_count": name for name in _HISTOGRAMS}
    for family in text_string_to_metric_families(text):
        for sample in family.samples:
            if sample.name == _ACTIVE_WORKERS:
                mode = sample.labels.get("disagg_mode", "")
                count = int(sample.value)
                if mode == "prefill":
                    snapshot.active_prefill = max(snapshot.active_prefill, count)
                elif mode == "decode":
                    snapshot.active_decode = max(snapshot.active_decode, count)
                elif mode == "mixed":
                    snapshot.active_mixed = max(snapshot.active_mixed, count)
                continue
            if sample.labels.get("outcome", "ok") != "ok":
                continue
            metric = wanted_sums.get(sample.name)
            index = 0
            if metric is None:
                metric = wanted_counts.get(sample.name)
                index = 1
            if metric is None:
                continue
            total = list(snapshot.totals_for(metric))
            total[index] += float(sample.value)
            snapshot.totals[metric] = (total[0], total[1])


def _window_average(current: Snapshot, previous: Snapshot, metric: str) -> float | None:
    """Mean of ``metric`` over the window, or None if the window is unusable.

    None means either "no observations in this window" (a legitimately idle
    interval) or "counters went backwards" (a server restarted); both are
    reasons to leave the deployment alone rather than guess.
    """
    cur_sum, cur_count = current.totals_for(metric)
    prev_sum, prev_count = previous.totals_for(metric)
    d_sum, d_count = cur_sum - prev_sum, cur_count - prev_count
    if d_sum < 0 or d_count < 0:
        return None
    if d_count == 0:
        return 0.0
    return d_sum / d_count


class MetricsSource:
    """Scrapes a set of Infera server ``/metrics`` endpoints and windows them."""

    def __init__(self, urls: list[str], *, timeout: float = 10.0) -> None:
        if not urls:
            raise ValueError("MetricsSource needs at least one /metrics URL")
        self._urls = list(urls)
        self._client = httpx.AsyncClient(timeout=timeout)
        self._previous: Snapshot | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def scrape(self) -> Snapshot:
        """Fetch and sum every configured endpoint. Unreachable replicas are
        logged and skipped, so one dead server doesn't blind the planner."""
        snapshot = Snapshot()
        for url in self._urls:
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("scrape of %s failed: %s: %s", url, type(exc).__name__, exc)
                continue
            try:
                parse_metrics_text(resp.text, snapshot)
            except ValueError as exc:
                logger.warning("could not parse metrics from %s: %s", url, exc)
                continue
            snapshot.scraped += 1
        return snapshot

    async def collect(self) -> LoadMetrics | None:
        """Scrape and difference against the previous round.

        Returns None on the first call (no window yet), when every replica was
        unreachable, or when the counters went backwards.
        """
        current = await self.scrape()
        if current.scraped == 0:
            logger.warning("no server /metrics endpoint could be scraped; skipping interval")
            return None

        previous, self._previous = self._previous, current
        if previous is None:
            logger.info("first scrape recorded; the first decision comes one interval from now")
            return None

        averages: dict[str, float] = {}
        for metric in (_TTFT, _ITL, _ISL, _OSL):
            value = _window_average(current, previous, metric)
            if value is None:
                logger.warning(
                    "%s went backwards (a server restarted?); skipping this interval", metric
                )
                return None
            averages[metric] = value

        _, prev_requests = previous.totals_for(_REQUESTS)
        _, cur_requests = current.totals_for(_REQUESTS)
        num_req = cur_requests - prev_requests
        if num_req < 0:
            logger.warning("request count went backwards; skipping this interval")
            return None

        return LoadMetrics(
            num_req=num_req,
            isl=averages[_ISL],
            osl=averages[_OSL],
            ttft=averages[_TTFT],
            itl=averages[_ITL],
            num_prefill=current.active_prefill,
            num_decode=current.active_decode,
            num_mixed=current.active_mixed,
        )
