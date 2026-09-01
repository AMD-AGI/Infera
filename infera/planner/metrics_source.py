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
``sum(endpoint deltas) / sum(endpoint count deltas)``. Baselines are retained
per URL: a restart is detected even when the rest of the fleet's counters keep
the aggregate increasing, and a temporarily missing endpoint resets the whole
baseline instead of injecting its lifetime counters when it returns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

_TTFT = "infera_time_to_first_token_seconds"
_ITL = "infera_inter_token_latency_seconds"
_ISL = "infera_input_sequence_tokens"
_OSL = "infera_output_sequence_tokens"
_ACTIVE_WORKERS = "infera_active_workers"

_HISTOGRAMS = (_TTFT, _ITL, _ISL, _OSL)


@dataclass
class Snapshot:
    """Fleet-summed cumulative values from one scrape round."""

    # metric name -> (sum, count)
    totals: dict[str, tuple[float, float]] = field(default_factory=dict)
    active_prefill: int = 0
    active_decode: int = 0
    active_mixed: int = 0

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
        """Whether the interval carries enough signal to size a fleet from."""
        return self.num_req > 0 and self.isl > 0 and self.osl > 0

    @property
    def has_latency(self) -> bool:
        """Whether the window recorded both policy latency signals.

        Both histograms are streaming-only -- a non-streaming reply arrives in
        one piece, so it has no observable first-token boundary -- and ITL
        additionally needs a reply of at least two tokens. Zero means
        unmeasured here, not infinitely fast.
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


def parse_metrics_text(
    text: str,
    snapshot: Snapshot,
    *,
    model: str | None = None,
    router: str | None = "disagg",
) -> None:
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
                if model is not None and sample.labels.get("model") != model:
                    continue
                mode = sample.labels.get("disagg_mode", "")
                count = int(sample.value)
                if mode == "prefill":
                    snapshot.active_prefill = max(snapshot.active_prefill, count)
                elif mode == "decode":
                    snapshot.active_decode = max(snapshot.active_decode, count)
                elif mode == "mixed":
                    snapshot.active_mixed = max(snapshot.active_mixed, count)
                continue
            metric = wanted_sums.get(sample.name)
            index = 0
            if metric is None:
                metric = wanted_counts.get(sample.name)
                index = 1
            if metric is None:
                continue
            if router is not None and sample.labels.get("router") != router:
                continue
            if model is not None and sample.labels.get("model") != model:
                continue
            total = list(snapshot.totals_for(metric))
            total[index] += float(sample.value)
            snapshot.totals[metric] = (total[0], total[1])


def _window_delta(current: Snapshot, previous: Snapshot, metric: str) -> tuple[float, float] | None:
    """``(sum, count)`` accrued over the window, or None if it is unusable.

    None means the counters went backwards, i.e. a server restarted, which is a
    reason to leave the deployment alone rather than guess.
    """
    cur_sum, cur_count = current.totals_for(metric)
    prev_sum, prev_count = previous.totals_for(metric)
    d_sum, d_count = cur_sum - prev_sum, cur_count - prev_count
    if d_sum < 0 or d_count < 0:
        return None
    return d_sum, d_count


class MetricsSource:
    """Scrapes a set of Infera server ``/metrics`` endpoints and windows them."""

    def __init__(
        self,
        urls: list[str],
        *,
        model: str | None = None,
        router: str | None = "disagg",
        timeout: float = 10.0,
    ) -> None:
        if not urls:
            raise ValueError("MetricsSource needs at least one /metrics URL")
        self._urls = list(urls)
        self._model = model
        self._router = router
        # Scrapes are minutes apart in production. Reusing an idle connection
        # races common 5-second server keep-alive timeouts, so open a fresh
        # connection for each low-frequency scrape.
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._previous: dict[str, Snapshot] | None = None

    async def aclose(self) -> None:
        await self._client.aclose()

    async def scrape(self) -> dict[str, Snapshot]:
        """Fetch every configured endpoint into its own cumulative snapshot."""
        snapshots: dict[str, Snapshot] = {}
        for url in self._urls:
            try:
                resp = await self._client.get(url)
                resp.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("scrape of %s failed: %s: %s", url, type(exc).__name__, exc)
                continue
            snapshot = Snapshot()
            try:
                parse_metrics_text(
                    resp.text,
                    snapshot,
                    model=self._model,
                    router=self._router,
                )
            except ValueError as exc:
                logger.warning("could not parse metrics from %s: %s", url, exc)
                continue
            snapshots[url] = snapshot
        return snapshots

    async def collect(self) -> LoadMetrics | None:
        """Scrape and difference against the previous round.

        Returns None on the first call (no window yet), when every replica was
        unreachable, or when the counters went backwards.
        """
        current = await self.scrape()
        if not current:
            logger.warning("no server /metrics endpoint could be scraped; skipping interval")
            return None

        previous, self._previous = self._previous, current
        if previous is None:
            logger.info("first scrape recorded; the first decision comes one interval from now")
            return None
        if current.keys() != previous.keys():
            logger.warning(
                "the set of scraped /metrics endpoints changed; resetting the baseline "
                "to avoid treating a returning endpoint's lifetime counters as one window"
            )
            return None

        averages: dict[str, float] = {}
        counts: dict[str, float] = {}
        for metric in _HISTOGRAMS:
            d_sum = 0.0
            d_count = 0.0
            for url, snapshot in current.items():
                delta = _window_delta(snapshot, previous[url], metric)
                if delta is None:
                    logger.warning(
                        "%s went backwards at %s (server restart?); skipping this interval",
                        metric,
                        url,
                    )
                    return None
                endpoint_sum, endpoint_count = delta
                d_sum += endpoint_sum
                d_count += endpoint_count
            counts[metric] = d_count
            averages[metric] = d_sum / d_count if d_count else 0.0

        snapshots = list(current.values())

        return LoadMetrics(
            # The OSL histogram's count, rather than the request counter, so the
            # request rate is over exactly the requests that contributed the
            # averages beside it. The request counter is committed at hand-off,
            # before a streaming reply can be disowned as failed, so a window
            # with partial failures would divide by a larger population than it
            # measured and overstate the token rate.
            num_req=counts[_OSL],
            isl=averages[_ISL],
            osl=averages[_OSL],
            ttft=averages[_TTFT],
            itl=averages[_ITL],
            num_prefill=max(snapshot.active_prefill for snapshot in snapshots),
            num_decode=max(snapshot.active_decode for snapshot in snapshots),
            num_mixed=max(snapshot.active_mixed for snapshot in snapshots),
        )
