###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The decision loop: from observed metrics to a replica count per pool.

Each adjustment interval corrects the performance model against what was
observed, assumes the next interval repeats that load, and solves each pool for
the replica count that meets its target. ``manual/features/sla_planner.md``
walks through the reasoning; the formulas are documented on the methods below.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections.abc import Awaitable, Callable

from infera.planner.args import PlannerArgs
from infera.planner.decision import ScalingDecision
from infera.planner.metrics_source import LoadMetrics, MetricsSource
from infera.planner.perf_model import PerfModel

logger = logging.getLogger(__name__)

_MS_PER_S = 1000.0
# Profiling and production TTFT both contain measurement noise. A target with
# less headroom than this fraction of the unqueued service time is too close to
# the hardware floor for the linear queue model to produce a trustworthy count.
_MIN_TTFT_HEADROOM_RATIO = 0.20
# Autoscaling recommendations should move toward the target rather than jump by
# orders of magnitude from one noisy window. This applies to both pools.
_MAX_SCALE_UP_RATIO = 4.0
DecisionHandler = Callable[[ScalingDecision], Awaitable[None]]


class SlaPlanner:
    """Holds the performance model and the periodic decision loop."""

    def __init__(
        self,
        args: PlannerArgs,
        perf_model: PerfModel,
        *,
        metrics_source: MetricsSource | None = None,
        on_decision: DecisionHandler | None = None,
    ) -> None:
        self.args = args
        self.perf = perf_model
        self._metrics_source = metrics_source
        # Optional seam for a future Kubernetes/Slurm actuator. The standalone
        # planner intentionally passes no handler and only logs decisions.
        self._on_decision = on_decision

        self.prefill_num_gpu = args.prefill_engine_num_gpu or perf_model.prefill_engine_num_gpu
        self.decode_num_gpu = args.decode_engine_num_gpu or perf_model.decode_engine_num_gpu

    # ------------------------------------------------------------------
    # Pure decision logic -- no I/O
    # ------------------------------------------------------------------

    def correction_factors(self, metrics: LoadMetrics) -> tuple[float, float] | None:
        """Observed-over-predicted latency for each phase.

        Profiling measures one request at a time on an idle engine, which
        production does not look like. Dividing what was observed by what the
        model predicted for the same workload absorbs the difference::

            prefill_correction = observed_ttft / predicted_ttft(isl)
            decode_correction  = observed_itl  / predicted_itl(concurrency, ctx)

        Prefill normally lands above 1.0, because real TTFT includes queueing
        the sweep never saw, and below 1.0 when prefix-cache hits mean the
        engine prefills fewer tokens than the prompt length implies. Decode
        should sit near 1.0; drift there usually means chunked prefill is
        stealing decode steps.

        Returns None when the model predicts a non-positive latency, which means
        the profiling data is degenerate and no correction can be derived.
        """
        predicted_ttft_ms = self.perf.prefill.interpolate_ttft(metrics.isl)
        if predicted_ttft_ms <= 0:
            logger.warning("profiled TTFT at isl=%.0f is not positive; skipping", metrics.isl)
            return None

        # Requests in flight per decode replica: arrival rate x residency.
        concurrency = (
            metrics.num_req
            / metrics.num_decode
            * metrics.request_duration
            / self.args.adjustment_interval
        )
        predicted_itl_ms = self.perf.decode.interpolate_itl(
            concurrency, metrics.isl + metrics.osl / 2
        )
        if predicted_itl_ms <= 0:
            logger.warning(
                "profiled ITL at concurrency=%.2f is not positive; skipping", concurrency
            )
            return None

        return (
            metrics.ttft * _MS_PER_S / predicted_ttft_ms,
            metrics.itl * _MS_PER_S / predicted_itl_ms,
        )

    def plan(self, metrics: LoadMetrics) -> ScalingDecision | None:
        """Produce the decision for the interval about to start.

        Returns None when the interval carries too little signal to act on: no
        traffic, no decode replicas to measure against, or a performance model
        that cannot be corrected.
        """
        if not metrics.has_traffic:
            logger.info(
                "no traffic in the last window (requests=%.0f, isl=%.0f, osl=%.0f); "
                "leaving the deployment alone",
                metrics.num_req,
                metrics.isl,
                metrics.osl,
            )
            return None

        if metrics.num_decode <= 0:
            logger.warning(
                "traffic observed but no decode replicas are registered; "
                "cannot calibrate the decode model"
            )
            return None

        if not metrics.has_latency:
            # Zero here means "never measured", not "instant": both corrections
            # would come out at 0, which collapses prefill demand to nothing and
            # loosens the ITL target to infinity -- the planner would size a busy
            # fleet down to one replica per pool and call it done.
            logger.warning(
                "traffic observed but no TTFT/ITL was recorded (ttft=%.0fms itl=%.1fms); "
                "TTFT needs a streaming reply and ITL a reply of at least two tokens, so "
                "a non-streaming workload leaves nothing to calibrate against and the "
                "deployment is left alone",
                metrics.ttft * _MS_PER_S,
                metrics.itl * _MS_PER_S,
            )
            return None

        corrections = self.correction_factors(metrics)
        if corrections is None:
            return None
        p_correction, d_correction = corrections

        # The minimal planner assumes the next interval repeats the last one.
        # A predictor can be added before this call later without changing the
        # sizing functions or the decision handler seam.
        num_prefill = self._plan_prefill(
            metrics.num_req, metrics.isl, p_correction, metrics.num_prefill
        )
        num_decode = self._plan_decode(metrics.num_req, metrics.isl, metrics.osl, d_correction)
        num_prefill = self._limit_scale_up(num_prefill, metrics.num_prefill, "prefill")
        num_decode = self._limit_scale_up(num_decode, metrics.num_decode, "decode")

        return ScalingDecision(
            num_prefill=max(1, num_prefill),
            num_decode=max(1, num_decode),
            observed_prefill=metrics.num_prefill,
            observed_decode=metrics.num_decode,
            prefill_correction=p_correction,
            decode_correction=d_correction,
            num_req=metrics.num_req,
            isl=metrics.isl,
            osl=metrics.osl,
        )

    @staticmethod
    def _limit_scale_up(desired: int, observed: int, role: str) -> int:
        """Bound one recommendation to a gradual, reviewable scale-up step."""
        maximum = math.ceil(max(1, observed) * _MAX_SCALE_UP_RATIO)
        if desired > maximum:
            logger.warning(
                "%s sizing asks for %d replicas from %d observed; limiting this decision "
                "to %d (%.0fx scale-up guardrail)",
                role,
                desired,
                observed,
                maximum,
                _MAX_SCALE_UP_RATIO,
            )
            return maximum
        return desired

    def _plan_prefill(
        self,
        next_num_req: float,
        next_isl: float,
        p_correction: float,
        observed_prefill: int,
    ) -> int:
        """Replicas needed to prefill the predicted token rate within the TTFT SLA::

            ceil(token_rate * min(1, p_correction) / thpt_per_gpu(isl) / gpus)

        ``min(1, ...)`` is deliberate: a correction above 1.0 means requests are
        queueing, and queueing is what adding replicas fixes, so inflating
        demand on top of it would count the same problem twice. Below 1.0 --
        prefix-cache hits shrinking the real prefill -- it scales demand down
        honestly.

        A pool sized for throughput alone can still miss the TTFT target by
        queueing, which is what :meth:`_prefill_for_ttft` covers; the larger of
        the two wins.
        """
        token_rate = next_num_req * next_isl / self.args.adjustment_interval
        demand = token_rate * min(1.0, p_correction)
        thpt_per_gpu = self.perf.prefill.interpolate_thpt_per_gpu(next_isl)
        for_throughput = math.ceil(demand / thpt_per_gpu / self.prefill_num_gpu)
        for_ttft = self._prefill_for_ttft(next_isl, p_correction, observed_prefill)
        if for_ttft > for_throughput:
            logger.info(
                "prefill sized by the TTFT target rather than throughput: %d replicas "
                "instead of %d, to hold TTFT at or under %.0fms",
                for_ttft,
                for_throughput,
                self.args.ttft_ms,
            )
        return max(for_throughput, for_ttft)

    def _prefill_for_ttft(self, next_isl: float, p_correction: float, observed_prefill: int) -> int:
        """Replicas needed to bring queueing down to the TTFT target.

        The observed TTFT splits along the correction factor::

            service_ms = profiled_ttft(isl) * min(1, p_correction)
            queue_ms   = profiled_ttft(isl) * max(0, p_correction - 1)

        Service is what prefilling costs once a request reaches an engine, and
        the remainder is time spent waiting for one. Adding replicas cannot
        touch the first and divides the second, so the count that fits the
        target is ``queue x replicas_now / headroom``.

        Returns 0 when the target imposes no requirement beyond throughput:
        nothing is queueing, or service time alone already overruns the target
        and no number of replicas would recover it.
        """
        profiled_ttft_ms = self.perf.prefill.interpolate_ttft(next_isl)
        service_ms = profiled_ttft_ms * min(1.0, p_correction)
        queue_ms = profiled_ttft_ms * max(0.0, p_correction - 1.0)
        if queue_ms <= 0.0:
            return 0

        headroom_ms = self.args.ttft_ms - service_ms
        if headroom_ms <= 0.0:
            logger.warning(
                "prefilling a %.0f-token prompt costs %.0fms on an unqueued engine, which "
                "already overruns the %.0fms TTFT target; replicas divide queueing, not "
                "service time, so prefill is sized for throughput alone",
                next_isl,
                service_ms,
                self.args.ttft_ms,
            )
            return 0
        minimum_headroom_ms = service_ms * _MIN_TTFT_HEADROOM_RATIO
        if headroom_ms < minimum_headroom_ms:
            logger.warning(
                "TTFT target %.1fms leaves only %.1fms queueing headroom above the "
                "%.1fms profiled service time, below the %.0f%% safety margin; the "
                "linear queue model is not reliable this close to the hardware floor, "
                "so prefill is sized for throughput alone",
                self.args.ttft_ms,
                headroom_ms,
                service_ms,
                _MIN_TTFT_HEADROOM_RATIO * 100,
            )
            return 0

        # A fleet reporting no prefill replicas cannot be divided by; treat it
        # as one so the ratio still means "relative to what is running".
        return math.ceil(queue_ms * max(1, observed_prefill) / headroom_ms)

    def _plan_decode(
        self, next_num_req: float, next_isl: float, next_osl: float, d_correction: float
    ) -> int:
        """Replicas needed to generate the predicted token rate within the ITL SLA::

            ceil(token_rate / thpt_per_gpu(itl_target / d_correction) / gpus)

        Tightening the ITL *target* by the correction factor and then asking the
        model for the throughput that fits is what makes this self-calibrating:
        a deployment running at twice the profiled ITL is aimed at half the
        target, and lands on the real one.
        """
        corrected_itl_ms = self.args.itl_ms / max(d_correction, 1e-6)
        context_length = next_isl + next_osl / 2
        thpt_per_gpu, achieved_itl_ms, kv_usage = self.perf.decode.find_best_throughput_per_gpu(
            corrected_itl_ms, context_length
        )
        if achieved_itl_ms > corrected_itl_ms:
            logger.warning(
                "even the lightest profiled decode load gives ITL=%.1fms > target %.1fms at "
                "context length %.0f; the ITL SLA is not reachable on this hardware",
                achieved_itl_ms,
                corrected_itl_ms,
                context_length,
            )
        else:
            logger.debug(
                "decode operating point: %.0f tok/s/gpu at %.0f%% kv usage (ITL %.1fms)",
                thpt_per_gpu,
                kv_usage * 100,
                achieved_itl_ms,
            )
        token_rate = next_num_req * next_osl / self.args.adjustment_interval
        return math.ceil(token_rate / thpt_per_gpu / self.decode_num_gpu)

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    async def tick(self) -> ScalingDecision | None:
        """Run one observe-decide-report cycle."""
        assert self._metrics_source is not None, "tick() needs a MetricsSource"
        metrics = await self._metrics_source.collect()
        if metrics is None:
            return None

        logger.info(
            "observed %.0f requests, isl=%.0f osl=%.0f, ttft=%.0fms itl=%.1fms, "
            "fleet prefill=%d decode=%d",
            metrics.num_req,
            metrics.isl,
            metrics.osl,
            metrics.ttft * _MS_PER_S,
            metrics.itl * _MS_PER_S,
            metrics.num_prefill,
            metrics.num_decode,
        )

        decision = self.plan(metrics)
        if decision is None:
            return None

        if not decision.changes_anything:
            logger.info(
                "fleet is already the right size (prefill=%d, decode=%d)",
                decision.num_prefill,
                decision.num_decode,
            )
            return decision

        logger.info("decision: %s", decision.summary())
        if self._on_decision is not None:
            await self._on_decision(decision)
        return decision

    async def run(self) -> None:
        """Tick forever, one adjustment interval apart.

        The first interval only records a baseline scrape -- the metrics are
        cumulative, so there is nothing to difference yet.
        """
        interval = self.args.adjustment_interval
        logger.info(
            "SLA planner started: model=%s ttft<=%.0fms itl<=%.1fms, interval %.0fs, "
            "%d GPU/prefill replica, %d GPU/decode replica",
            self.args.model,
            self.args.ttft_ms,
            self.args.itl_ms,
            interval,
            self.prefill_num_gpu,
            self.decode_num_gpu,
        )
        while True:
            started = time.monotonic()
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                # One bad interval must not take the planner down; the next
                # scrape re-establishes the window.
                logger.exception("adjustment interval failed; retrying next interval")
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, interval - elapsed))
