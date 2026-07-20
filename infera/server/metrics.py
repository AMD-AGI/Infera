###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Prometheus metrics for Infera.

All metric definitions live in this module so there's exactly one
place to look for "what does the server expose to ops". The router
and policy code calls these helpers; the FastAPI app mounts the
exposition endpoint at /metrics.

Naming follows the Prometheus convention:
    infera_<subsystem>_<thing>_<unit>

Labels are kept low-cardinality. ``worker_id`` is OK (fleet is bounded);
free-form fields like model name and request_id are NOT label values.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from prometheus_client.exposition import CONTENT_TYPE_LATEST

REGISTRY = CollectorRegistry()


# ----------------------------------------------------------------------
# Routing decisions
# ----------------------------------------------------------------------

router_picks_total = Counter(
    "infera_router_picks_total",
    "Total number of pick() decisions made by the router, per role and worker.",
    labelnames=("role", "worker_id"),  # role ∈ {prefill, decode, mixed}
    registry=REGISTRY,
)

router_pick_cache_hits = Histogram(
    "infera_router_pick_cache_hits",
    "Number of cache blocks the picked worker already had (chained-prefix hits).",
    labelnames=("role",),
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, float("inf")),
    registry=REGISTRY,
)

router_pick_request_blocks = Histogram(
    "infera_router_pick_request_blocks",
    "Total request block count seen at pick time.",
    labelnames=("role",),
    buckets=(0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, float("inf")),
    registry=REGISTRY,
)


# ----------------------------------------------------------------------
# Request lifecycle
# ----------------------------------------------------------------------

request_duration_seconds = Histogram(
    "infera_request_duration_seconds",
    "End-to-end server-observed request latency (server.pick + worker round-trip).",
    labelnames=("router", "outcome"),  # router ∈ {mixed, disagg}; outcome ∈ {ok, 5xx, 4xx, error}
    buckets=(
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        float("inf"),
    ),
    registry=REGISTRY,
)

request_inflight = Gauge(
    "infera_request_inflight",
    "Currently in-flight requests at the server.",
    labelnames=("router",),
    registry=REGISTRY,
)


# ----------------------------------------------------------------------
# PD-disaggregation specifics
# ----------------------------------------------------------------------

pd_dispatch_duration_seconds = Histogram(
    "infera_pd_dispatch_duration_seconds",
    "Per-worker duration of a PD-dispatched request (P or D leg from server "
    "POST to response complete). Roughly: P leg = prefill + KV push start; "
    "D leg = KV pull + decode generation. The gap between P and D "
    "completion times approximates KV transfer latency.",
    labelnames=("leg", "worker_id"),  # leg ∈ {prefill, decode}
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float("inf")),
    registry=REGISTRY,
)

pd_bootstrap_failures_total = Counter(
    "infera_pd_bootstrap_failures_total",
    "PD bootstrap protocol failures (missing bootstrap_addr, P unreachable, etc.).",
    labelnames=("reason",),  # reason ∈ {missing_bootstrap_addr, p_unreachable, d_unreachable,
    #                decode_5xx, decode_stream_broken, prefill_exception, prefill_5xx, ...}
    registry=REGISTRY,
)


# ----------------------------------------------------------------------
# Worker pool state
# ----------------------------------------------------------------------

active_workers = Gauge(
    "infera_active_workers",
    "Number of workers in ACTIVE status, by disagg_mode.",
    labelnames=("disagg_mode",),
    registry=REGISTRY,
)


# ----------------------------------------------------------------------
# KV-aware policy internals
# ----------------------------------------------------------------------

policy_active_blocks = Gauge(
    "infera_policy_active_blocks",
    "Per-worker count of distinct in-flight block hashes — the load term in "
    "KvEventAwarePolicy's cost function. Lower = less loaded by the policy's view.",
    labelnames=("worker_id",),
    registry=REGISTRY,
)

policy_cache_view_size = Gauge(
    "infera_policy_cache_view_size",
    "Per-worker count of cached blocks in the router-side KvEventClient view. "
    "Reflects what BlockStored / BlockRemoved events have applied so far.",
    labelnames=("worker_id",),
    registry=REGISTRY,
)


# ----------------------------------------------------------------------
# Client cache-control hints (retention)
# ----------------------------------------------------------------------

cache_control_seen_total = Counter(
    "infera_cache_control_seen_total",
    "Count of requests by retention level observed in the request body "
    "(parsed from Anthropic cache_control / OpenAI prompt_cache_retention). "
    "Lets ops correlate workload mix with cache hit rate.",
    labelnames=("retention",),  # ∈ {none, short, long}
    registry=REGISTRY,
)

cache_locality_skipped_total = Counter(
    "infera_cache_locality_skipped_total",
    "Count of routing decisions where cache locality was intentionally "
    "ignored (overlap_weight forced to 0). Most common reason today is "
    "`multimodal` — see Phase 4.7(b): until router + engine adopt MM-aware "
    "hashing, vision/audio requests fall back to pure load balance to "
    "avoid wrong-KV reuse from same-text-different-image collisions.",
    labelnames=("reason",),
    registry=REGISTRY,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def record_pick(*, role: str, worker_id: str, cache_hits: int, request_blocks: int) -> None:
    """Single entry point the router calls on every policy.pick() decision."""
    router_picks_total.labels(role=role, worker_id=worker_id).inc()
    router_pick_cache_hits.labels(role=role).observe(cache_hits)
    router_pick_request_blocks.labels(role=role).observe(request_blocks)


@contextmanager
def track_request(router: str):
    """Context manager that wraps a request's server-side lifetime to
    populate `request_duration_seconds` (with outcome) and the in-flight
    gauge. Use like::

        with track_request(router="mixed") as obs:
            resp = await dispatch(...)
            obs["outcome"] = "ok" if resp.status_code < 400 else f"{resp.status_code // 100}xx"
    """
    state = {"outcome": "error"}
    request_inflight.labels(router=router).inc()
    start = time.perf_counter()
    try:
        yield state
    finally:
        request_inflight.labels(router=router).dec()
        request_duration_seconds.labels(router=router, outcome=state["outcome"]).observe(
            time.perf_counter() - start
        )


@contextmanager
def track_pd_leg(*, leg: str, worker_id: str):
    """Time a single PD leg (P or D). Used inside `DisaggRouter`.

    The gap between the P leg's end-time and the D leg's first-token
    time is the closest proxy we have to KV-transfer latency without
    hooking inside SGLang.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        pd_dispatch_duration_seconds.labels(leg=leg, worker_id=worker_id).observe(
            time.perf_counter() - start
        )


def render_metrics() -> tuple[bytes, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
