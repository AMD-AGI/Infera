###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Translate one observed workload window into pool capacity budgets."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from infera.planner.args import PlannerArgs
from infera.planner.metrics_source import LoadMetrics
from infera.planner.perf_model import PerfModel, PrefillPoint

logger = logging.getLogger(__name__)

_MILLISECONDS = 1000.0
_MIN_QUEUE_HEADROOM = 0.20
_MAX_GROWTH_STEP = 4.0


@dataclass(frozen=True)
class LatencyRatios:
    """Observed latency divided by the offline envelope at the same workload."""

    prompt: float
    generation: float


@dataclass(frozen=True)
class CapacityRecommendation:
    """Replica budgets and the evidence used to derive them."""

    prefill_replicas: int
    decode_replicas: int
    latency: LatencyRatios


class CapacityPolicy:
    """Pure capacity policy; contains no scraping, sleeping, or actuation."""

    def __init__(self, args: PlannerArgs, model: PerfModel) -> None:
        self._args = args
        self._model = model
        self._prefill_gpus = args.prefill_engine_num_gpu or model.prefill_engine_num_gpu
        self._decode_gpus = args.decode_engine_num_gpu or model.decode_engine_num_gpu

    @property
    def prefill_gpus(self) -> int:
        return self._prefill_gpus

    @property
    def decode_gpus(self) -> int:
        return self._decode_gpus

    def recommend(self, window: LoadMetrics) -> CapacityRecommendation | None:
        prompt_point = self._model.prompt_capacity(window.isl)
        mean_context = window.isl + window.osl / 2.0
        in_flight_per_decode = (
            window.num_req
            * window.request_duration
            / self._args.adjustment_interval
            / window.num_decode
        )
        generation_point = self._model.generation_point(
            in_flight_per_decode,
            mean_context,
        )
        if prompt_point.latency_ms <= 0 or generation_point.latency_ms <= 0:
            logger.warning("profile envelope contains a non-positive latency; window ignored")
            return None

        ratios = LatencyRatios(
            prompt=window.ttft * _MILLISECONDS / prompt_point.latency_ms,
            generation=window.itl * _MILLISECONDS / generation_point.latency_ms,
        )
        prefill = self._prefill_budget(window, prompt_point, ratios.prompt)
        decode = self._decode_budget(window, mean_context, ratios.generation)
        return CapacityRecommendation(
            prefill_replicas=self._bounded_growth(
                prefill,
                window.num_prefill,
                "prefill",
            ),
            decode_replicas=self._bounded_growth(
                decode,
                window.num_decode,
                "decode",
            ),
            latency=ratios,
        )

    def _prefill_budget(
        self,
        window: LoadMetrics,
        profile: PrefillPoint,
        latency_ratio: float,
    ) -> int:
        arriving_tokens_per_second = window.num_req * window.isl / self._args.adjustment_interval
        cache_adjusted_rate = arriving_tokens_per_second * min(1.0, latency_ratio)
        measured_replica_capacity = profile.tokens_per_second_per_gpu * self._prefill_gpus
        throughput_budget = math.ceil(cache_adjusted_rate / measured_replica_capacity)
        queue_budget = self._queue_budget(
            profile,
            latency_ratio,
            window.num_prefill,
        )
        return max(throughput_budget, queue_budget)

    def _queue_budget(
        self,
        profile: PrefillPoint,
        latency_ratio: float,
        current_replicas: int,
    ) -> int:
        service_ms = profile.latency_ms * min(1.0, latency_ratio)
        waiting_ms = profile.latency_ms * max(0.0, latency_ratio - 1.0)
        if waiting_ms <= 0:
            return 0

        available_ms = self._args.ttft_ms - service_ms
        if available_ms <= 0:
            logger.warning(
                "profiled prompt service time %.1fms is already above the %.1fms target; "
                "using throughput capacity only",
                service_ms,
                self._args.ttft_ms,
            )
            return 0
        if available_ms < service_ms * _MIN_QUEUE_HEADROOM:
            logger.warning(
                "TTFT target leaves %.1fms queue allowance above %.1fms service time; "
                "the allowance is too small for a stable capacity estimate",
                available_ms,
                service_ms,
            )
            return 0
        return math.ceil(waiting_ms * max(1, current_replicas) / available_ms)

    def _decode_budget(
        self,
        window: LoadMetrics,
        mean_context: float,
        latency_ratio: float,
    ) -> int:
        effective_budget_ms = self._args.itl_ms / max(latency_ratio, 1e-6)
        point = self._model.generation_capacity(effective_budget_ms, mean_context)
        if point.latency_ms > effective_budget_ms:
            logger.warning(
                "generation latency floor %.1fms is above the effective %.1fms budget "
                "at mean context %.0f",
                point.latency_ms,
                effective_budget_ms,
                mean_context,
            )
        generated_tokens_per_second = window.num_req * window.osl / self._args.adjustment_interval
        measured_replica_capacity = point.tokens_per_second_per_gpu * self._decode_gpus
        return math.ceil(generated_tokens_per_second / measured_replica_capacity)

    @staticmethod
    def _bounded_growth(wanted: int, observed: int, pool: str) -> int:
        ceiling = math.ceil(max(1, observed) * _MAX_GROWTH_STEP)
        if wanted <= ceiling:
            return max(1, wanted)
        logger.warning(
            "%s capacity requires %d replicas from %d observed; this decision is capped at %d",
            pool,
            wanted,
            observed,
            ceiling,
        )
        return ceiling
