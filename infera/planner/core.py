###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Observe, budget, and publish loop for the standalone planner process."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from infera.planner.args import PlannerArgs
from infera.planner.capacity import CapacityPolicy
from infera.planner.decision import ScalingDecision
from infera.planner.metrics_source import LoadMetrics, MetricsSource
from infera.planner.perf_model import PerfModel

logger = logging.getLogger(__name__)

DecisionHandler = Callable[[ScalingDecision], Awaitable[None]]


class SlaPlanner:
    """Coordinates observations and policy without entering the request path."""

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
        self._on_decision = on_decision
        self._policy = CapacityPolicy(args, perf_model)
        self.prefill_num_gpu = self._policy.prefill_gpus
        self.decode_num_gpu = self._policy.decode_gpus

    def plan(self, metrics: LoadMetrics) -> ScalingDecision | None:
        """Turn one complete observation window into target pool sizes."""
        if not metrics.has_traffic:
            logger.info(
                "window has no usable workload (requests=%.0f, isl=%.0f, osl=%.0f)",
                metrics.num_req,
                metrics.isl,
                metrics.osl,
            )
            return None
        if metrics.num_decode <= 0:
            logger.warning("generation traffic exists but no decode replica is observable")
            return None
        if not metrics.has_latency:
            logger.warning(
                "window has traffic but lacks streaming latency samples (ttft=%.1fms, itl=%.1fms)",
                metrics.ttft * 1000.0,
                metrics.itl * 1000.0,
            )
            return None

        recommendation = self._policy.recommend(metrics)
        if recommendation is None:
            return None
        return ScalingDecision(
            num_prefill=recommendation.prefill_replicas,
            num_decode=recommendation.decode_replicas,
            observed_prefill=metrics.num_prefill,
            observed_decode=metrics.num_decode,
            prefill_latency_ratio=recommendation.latency.prompt,
            decode_latency_ratio=recommendation.latency.generation,
            num_req=metrics.num_req,
            isl=metrics.isl,
            osl=metrics.osl,
        )

    async def tick(self) -> ScalingDecision | None:
        """Collect and publish at most one decision."""
        if self._metrics_source is None:
            raise RuntimeError("tick() requires a metrics source")
        metrics = await self._metrics_source.collect()
        if metrics is None:
            return None

        logger.info(
            "window: requests=%.0f isl=%.0f osl=%.0f ttft=%.0fms itl=%.1fms; "
            "pools prefill=%d decode=%d",
            metrics.num_req,
            metrics.isl,
            metrics.osl,
            metrics.ttft * 1000.0,
            metrics.itl * 1000.0,
            metrics.num_prefill,
            metrics.num_decode,
        )
        decision = self.plan(metrics)
        if decision is None:
            return None
        if not decision.changes_anything:
            logger.info(
                "current pools already satisfy the capacity budget (prefill=%d, decode=%d)",
                decision.num_prefill,
                decision.num_decode,
            )
            return decision

        logger.info("capacity decision: %s", decision.summary())
        if self._on_decision is not None:
            await self._on_decision(decision)
        return decision

    async def run(self) -> None:
        """Run observation windows until the process is cancelled."""
        interval = self.args.adjustment_interval
        logger.info(
            "planner ready: model=%s ttft<=%.0fms itl<=%.1fms window=%.0fs; "
            "gpus/replica prefill=%d decode=%d",
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
                logger.exception("planning window failed; the next window will retry")
            elapsed = time.monotonic() - started
            await asyncio.sleep(max(0.0, interval - elapsed))
