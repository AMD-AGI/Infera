###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""What the planner itself exposes on /metrics.

Scaling decisions are otherwise only visible in logs, which is awkward when a
deployment has been oscillating for an hour and you want to see the shape of it.
The correction factors are the most useful series here: sustained drift away
from 1.0 means the profiling data no longer describes the deployment, and every
decision built on it is suspect.

Kept in its own registry so the planner's metrics never collide with the
server's, which lives in :mod:`infera.server.metrics`.
"""

from __future__ import annotations

import logging

from prometheus_client import CollectorRegistry, Counter, Gauge, start_http_server

from infera.planner.decision import ScalingDecision
from infera.planner.metrics_source import LoadMetrics

logger = logging.getLogger(__name__)

REGISTRY = CollectorRegistry()

observed_ttft_seconds = Gauge(
    "infera_planner_observed_ttft_seconds",
    "Mean time to first token over the last observation window.",
    registry=REGISTRY,
)

observed_itl_seconds = Gauge(
    "infera_planner_observed_itl_seconds",
    "Mean inter-token latency over the last observation window.",
    registry=REGISTRY,
)

observed_sequence_tokens = Gauge(
    "infera_planner_observed_sequence_tokens",
    "Mean sequence length over the last observation window, by direction.",
    labelnames=("direction",),  # input | output
    registry=REGISTRY,
)

observed_requests = Gauge(
    "infera_planner_observed_requests",
    "Requests completed during the last observation window.",
    registry=REGISTRY,
)

correction_factor = Gauge(
    "infera_planner_correction_factor",
    "Observed latency divided by what the profiling data predicted. Prefill "
    "normally exceeds 1.0 (queueing adds to TTFT); decode should sit near 1.0. "
    "Sustained drift means the profiling data is stale.",
    labelnames=("phase",),  # prefill | decode
    registry=REGISTRY,
)

desired_replicas = Gauge(
    "infera_planner_desired_replicas",
    "Replica count the planner most recently asked for, by role.",
    labelnames=("role",),  # prefill | decode
    registry=REGISTRY,
)

decisions_total = Counter(
    "infera_planner_decisions_total",
    "Scaling decisions produced, split by whether they changed the deployment.",
    labelnames=("outcome",),  # applied | unchanged
    registry=REGISTRY,
)

intervals_skipped_total = Counter(
    "infera_planner_intervals_skipped_total",
    "Observation windows the planner declined to act on.",
    labelnames=("reason",),  # no_metrics | no_traffic | no_decode_workers | model_error
    registry=REGISTRY,
)

gpu_budget_exceeded_total = Counter(
    "infera_planner_gpu_budget_exceeded_total",
    "Decisions cut down to fit --max-gpu-budget. A non-zero rate here means "
    "the SLA is not reachable within the GPUs the planner is allowed to use.",
    registry=REGISTRY,
)


def record_observation(metrics: LoadMetrics) -> None:
    observed_ttft_seconds.set(metrics.ttft)
    observed_itl_seconds.set(metrics.itl)
    observed_sequence_tokens.labels(direction="input").set(metrics.isl)
    observed_sequence_tokens.labels(direction="output").set(metrics.osl)
    observed_requests.set(metrics.num_req)


def record_decision(decision: ScalingDecision) -> None:
    correction_factor.labels(phase="prefill").set(decision.prefill_correction)
    correction_factor.labels(phase="decode").set(decision.decode_correction)
    desired_replicas.labels(role="prefill").set(decision.num_prefill)
    desired_replicas.labels(role="decode").set(decision.num_decode)
    decisions_total.labels(outcome="applied" if decision.changes_anything else "unchanged").inc()
    if decision.gpu_budget_exceeded:
        gpu_budget_exceeded_total.inc()


def serve(port: int) -> None:
    """Start the exposition endpoint. A port of 0 disables it."""
    if port <= 0:
        return
    try:
        start_http_server(port, registry=REGISTRY)
    except OSError as exc:
        # Not fatal: losing observability is better than refusing to plan.
        logger.warning("could not serve planner metrics on port %d: %s", port, exc)
        return
    logger.info("planner metrics on :%d/metrics", port)
