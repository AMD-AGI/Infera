###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The scaling loop: from observed metrics to a replica count per pool.

Each adjustment interval runs four steps.

**1. Correct the performance model.** Profiling measures one request at a time
on an idle engine; production does not look like that. Comparing what was
observed against what the model predicted for the same workload gives a
correction factor per phase:

    prefill_correction = observed_ttft / predicted_ttft
    decode_correction  = observed_itl  / predicted_itl

Prefill normally lands above 1.0, because real TTFT includes queueing that the
profiling sweep never saw, and can land below 1.0 when prefix-cache hits mean
the engine prefills fewer tokens than the prompt length implies. Decode should
sit near 1.0; drift there usually means chunked prefill is stealing decode
steps.

**2. Forecast the next interval.** Scaling on the interval that just ended
always arrives one interval late, which on a rising ramp is exactly when the
SLA breaks. See :mod:`infera.planner.predictor`.

**3. Solve for replicas.** Prefill is single-batched, so its correction factor
scales throughput demand linearly. Decode is a batched steady state, so the
correction is applied to the ITL *target* instead and the model is inverted to
find the highest per-GPU throughput that still fits:

    prefill: ceil(token_rate * min(1, p_correction) / thpt_per_gpu / gpus)
    decode:  ceil(token_rate / thpt_per_gpu(itl_target / d_correction) / gpus)

``min(1, ...)`` on the prefill correction is deliberate: a correction above 1.0
means requests are queueing, and queueing is what adding replicas fixes, so
inflating demand on top of it would double-count.

**4. Clamp.** ``--min-endpoint`` keeps both pools alive, and
``--max-gpu-budget`` caps the total. Hitting the budget means the SLA is not
reachable with the GPUs on hand, which the planner records rather than hides.

The algorithm is adapted from NVIDIA Dynamo's SLA planner (Apache-2.0); this is
an independent implementation.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time

from infera.planner import planner_metrics
from infera.planner.args import PlannerArgs
from infera.planner.connectors.base import PlannerConnector
from infera.planner.decision import ScalingDecision
from infera.planner.metrics_source import LoadMetrics, MetricsSource
from infera.planner.perf_model import PerfModel
from infera.planner.predictor import build_predictor

logger = logging.getLogger(__name__)

_MS_PER_S = 1000.0


class SlaPlanner:
    """Holds the performance model, the load predictors, and the control loop."""

    def __init__(
        self,
        args: PlannerArgs,
        perf_model: PerfModel,
        *,
        metrics_source: MetricsSource | None = None,
        connector: PlannerConnector | None = None,
    ) -> None:
        self.args = args
        self.perf = perf_model
        self._metrics_source = metrics_source
        self._connector = connector

        self.prefill_num_gpu = args.prefill_engine_num_gpu or perf_model.prefill_engine_num_gpu
        self.decode_num_gpu = args.decode_engine_num_gpu or perf_model.decode_engine_num_gpu

        window = args.predictor_window
        self._num_req_predictor = build_predictor(args.load_predictor, window_size=window)
        self._isl_predictor = build_predictor(args.load_predictor, window_size=window)
        self._osl_predictor = build_predictor(args.load_predictor, window_size=window)

    # ------------------------------------------------------------------
    # Pure decision logic -- no I/O, so it can be exercised directly
    # ------------------------------------------------------------------

    def observe(self, metrics: LoadMetrics) -> None:
        """Feed one interval's observations to the load predictors."""
        self._num_req_predictor.add_data_point(metrics.num_req)
        self._isl_predictor.add_data_point(metrics.isl)
        self._osl_predictor.add_data_point(metrics.osl)

    def correction_factors(self, metrics: LoadMetrics) -> tuple[float, float] | None:
        """Observed-over-predicted latency for each phase.

        Returns None when the model predicts a non-positive latency, which means
        the profiling data is degenerate and no correction can be derived from
        it.
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
            planner_metrics.intervals_skipped_total.labels(reason="no_traffic").inc()
            return None

        if metrics.num_decode <= 0:
            logger.warning(
                "traffic observed but no decode replicas are registered; "
                "cannot calibrate the decode model"
            )
            planner_metrics.intervals_skipped_total.labels(reason="no_decode_workers").inc()
            return None

        corrections = self.correction_factors(metrics)
        if corrections is None:
            planner_metrics.intervals_skipped_total.labels(reason="model_error").inc()
            return None
        p_correction, d_correction = corrections

        next_num_req = self._num_req_predictor.predict_next()
        next_isl = self._isl_predictor.predict_next()
        next_osl = self._osl_predictor.predict_next()

        num_prefill = self._plan_prefill(next_num_req, next_isl, p_correction)
        num_decode = self._plan_decode(next_num_req, next_isl, next_osl, d_correction)

        num_prefill = max(num_prefill, self.args.min_endpoint)
        num_decode = max(num_decode, self.args.min_endpoint)
        num_prefill, num_decode, over_budget = self._apply_gpu_budget(num_prefill, num_decode)

        return ScalingDecision(
            num_prefill=num_prefill,
            num_decode=num_decode,
            observed_prefill=metrics.num_prefill,
            observed_decode=metrics.num_decode,
            prefill_correction=p_correction,
            decode_correction=d_correction,
            predicted_num_req=next_num_req,
            predicted_isl=next_isl,
            predicted_osl=next_osl,
            gpu_budget_exceeded=over_budget,
        )

    def _plan_prefill(self, next_num_req: float, next_isl: float, p_correction: float) -> int:
        """Replicas needed to prefill the predicted token rate.

        The correction is capped at 1.0: above that, TTFT is inflated by
        queueing, and more replicas are the cure rather than a reason to ask for
        even more work. Below 1.0 -- prefix-cache hits shrinking the real prefill
        -- it scales demand down honestly.
        """
        token_rate = next_num_req * next_isl / self.args.adjustment_interval
        demand = token_rate * min(1.0, p_correction)
        thpt_per_gpu = self.perf.prefill.interpolate_thpt_per_gpu(next_isl)
        return math.ceil(demand / thpt_per_gpu / self.prefill_num_gpu)

    def _plan_decode(
        self, next_num_req: float, next_isl: float, next_osl: float, d_correction: float
    ) -> int:
        """Replicas needed to generate the predicted token rate within the ITL SLA.

        Tightening the ITL target by the correction factor and then asking the
        model for the throughput that fits is what makes this self-calibrating:
        if the deployment runs slower than profiled, the planner aims at a
        stricter target and lands on the real one.
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

    def _apply_gpu_budget(self, num_prefill: int, num_decode: int) -> tuple[int, int, bool]:
        """Shrink both pools proportionally until they fit ``--max-gpu-budget``.

        Prefill is scaled first and decode takes whatever GPUs remain, so the
        rounding never overshoots the budget. Both pools still respect
        ``--min-endpoint``, which can itself exceed the budget on a
        deliberately undersized deployment -- the planner reports that rather
        than scaling a pool to zero.
        """
        budget = self.args.max_gpu_budget
        required = num_prefill * self.prefill_num_gpu + num_decode * self.decode_num_gpu
        if required <= budget:
            return num_prefill, num_decode, False

        scale = budget / required
        capped_prefill = max(self.args.min_endpoint, round(num_prefill * scale))
        remaining = budget - capped_prefill * self.prefill_num_gpu
        capped_decode = max(self.args.min_endpoint, remaining // self.decode_num_gpu)
        logger.warning(
            "SLA needs %d GPUs but the budget is %d; cutting prefill %d->%d, decode %d->%d "
            "(the SLA will not be met)",
            required,
            budget,
            num_prefill,
            capped_prefill,
            num_decode,
            capped_decode,
        )
        return capped_prefill, int(capped_decode), True

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    async def tick(self) -> ScalingDecision | None:
        """Run one observe-decide-apply cycle."""
        assert self._metrics_source is not None, "tick() needs a MetricsSource"
        metrics = await self._metrics_source.collect()
        if metrics is None:
            planner_metrics.intervals_skipped_total.labels(reason="no_metrics").inc()
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
        planner_metrics.record_observation(metrics)

        self.observe(metrics)
        decision = self.plan(metrics)
        if decision is None:
            return None

        planner_metrics.record_decision(decision)
        if not decision.changes_anything:
            logger.info(
                "no scaling needed (prefill=%d, decode=%d)",
                decision.num_prefill,
                decision.num_decode,
            )
            return decision

        if self._connector is not None:
            await self._connector.apply(decision)
        return decision

    async def run(self) -> None:
        """Tick forever, one adjustment interval apart.

        The first interval only records a baseline scrape -- the metrics are
        cumulative, so there is nothing to difference yet.
        """
        interval = self.args.adjustment_interval
        logger.info(
            "SLA planner started: ttft<=%.0fms itl<=%.1fms, interval %.0fs, "
            "%d GPU budget, %d GPU/prefill replica, %d GPU/decode replica",
            self.args.ttft_ms,
            self.args.itl_ms,
            interval,
            self.args.max_gpu_budget,
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
