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

import json
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
# SLA signals (consumed by infera.planner)
# ----------------------------------------------------------------------
# The SLA planner reads only the _sum / _count of these four histograms and
# window-differences them, so the bucket layout is for humans/dashboards.

time_to_first_token_seconds = Histogram(
    "infera_time_to_first_token_seconds",
    "Server-observed time from dispatch to the first token of the reply. "
    "For PD-disaggregated requests this spans prefill + KV transfer + the "
    "decode engine's first forward pass.",
    labelnames=("router",),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf")),
    registry=REGISTRY,
)

inter_token_latency_seconds = Histogram(
    "infera_inter_token_latency_seconds",
    "Mean per-request inter-token latency: (total stream time - TTFT) spread "
    "over the generated tokens. Only observed for requests that produced at "
    "least two output tokens.",
    labelnames=("router",),
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 1.0, float("inf")),
    registry=REGISTRY,
)

input_sequence_tokens = Histogram(
    "infera_input_sequence_tokens",
    "Prompt length in tokens (ISL). Exact when the engine reports "
    "`usage.prompt_tokens`; otherwise estimated from the router's KV block "
    "count, which rounds up to a block boundary.",
    labelnames=("router",),
    buckets=(64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, float("inf")),
    registry=REGISTRY,
)

output_sequence_tokens = Histogram(
    "infera_output_sequence_tokens",
    "Generated length in tokens (OSL). Exact when the engine reports "
    "`usage.completion_tokens`; otherwise counted as SSE data frames, which "
    "assumes one token per frame.",
    labelnames=("router",),
    buckets=(4, 16, 64, 256, 1024, 4096, 16384, 65536, float("inf")),
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


worker_breaker_state = Gauge(
    "infera_router_worker_breaker_state",
    "Router-side circuit breaker per worker: 0=closed, 1=half_open, 2=open. "
    "Non-zero means the router is routing around a worker that discovery still "
    "reports ACTIVE — the gap this metric exists to make visible.",
    labelnames=("worker_id",),
    registry=REGISTRY,
)

worker_breaker_trips_total = Counter(
    "infera_router_worker_breaker_trips_total",
    "Times a worker's breaker has opened. A worker tripping repeatedly while "
    "staying ACTIVE is broken for inference but healthy to the platform.",
    labelnames=("worker_id",),
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


_SSE_DATA = b"data:"
_SSE_DONE = b"data: [DONE]"
# An SSE frame that grows past this without a newline is not a token frame
# (or the peer is misbehaving); drop the partial rather than buffer forever.
_MAX_PARTIAL_FRAME = 1 << 16


class RequestObserver(dict):
    """Per-request accumulator for the SLA signals the planner consumes.

    Subclasses ``dict`` so the established ``obs["outcome"] = ...`` idiom in
    the routers keeps working untouched.

    Streaming responses outlive the ``track_request`` block — ``dispatch``
    returns a ``StreamingResponse`` and the generator runs afterwards — so the
    SLA histograms are emitted by :meth:`close` rather than at context exit.
    A generator that owns the token stream calls :meth:`claim_stream` and then
    :meth:`close` from its own ``finally``.
    """

    def __init__(self, router: str) -> None:
        super().__init__(outcome="error")
        self._router = router
        self._start = time.perf_counter()
        self._ttft: float | None = None
        self._isl: int | None = None
        self._osl: int | None = None
        self._frames = 0
        self._partial = b""
        self._deferred = False
        self._closed = False

    @property
    def deferred(self) -> bool:
        """True once a streaming generator took over responsibility for close()."""
        return self._deferred

    def claim_stream(self) -> None:
        self._deferred = True

    def first_token(self) -> None:
        """Mark the arrival of the first reply token. Idempotent."""
        if self._ttft is None:
            self._ttft = time.perf_counter() - self._start

    def set_input_tokens(self, n: int) -> None:
        if n > 0:
            self._isl = int(n)

    def set_output_tokens(self, n: int) -> None:
        """Record an exact OSL, overriding any SSE frame count."""
        if n > 0:
            self._osl = int(n)

    def observe_blocks(self, blocks: list[int] | None, block_size: int | None) -> None:
        """Estimate ISL from the router's KV block list, used when the engine
        reply carries no ``usage`` (the streaming default). Rounds up to a
        block boundary, and is a no-op for stateless policies that return no
        blocks."""
        if self._isl is None and blocks and block_size:
            self._isl = len(blocks) * int(block_size)

    def observe_usage(self, payload: object) -> None:
        """Pull exact ISL/OSL out of an OpenAI-shaped ``usage`` object."""
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        if isinstance(prompt, int):
            self.set_input_tokens(prompt)
        if isinstance(completion, int):
            self.set_output_tokens(completion)

    def observe_stream_chunk(self, chunk: bytes) -> None:
        """Count complete SSE data frames in a raw stream chunk as tokens.

        Frames are split on newlines and the trailing partial is carried over,
        so a chunk boundary never double-counts. ``[DONE]`` and any trailing
        usage frame are excluded, but the one-frame-per-token assumption still
        makes this an estimate; ``set_output_tokens`` supersedes it when the
        engine reports usage.
        """
        if not chunk:
            return
        self.first_token()
        buf = self._partial + chunk
        frames = buf.split(b"\n")
        tail = frames.pop()
        self._partial = tail if len(tail) <= _MAX_PARTIAL_FRAME else b""
        for frame in frames:
            frame = frame.strip()
            if frame.startswith(_SSE_DATA) and not frame.startswith(_SSE_DONE):
                self._frames += 1
                if b'"usage"' in frame:
                    # A usage-only frame carries no token; it also gives us the
                    # exact counts, which win over this whole estimate.
                    self._frames -= 1
                    try:
                        self.observe_usage(json.loads(frame[len(_SSE_DATA) :]))
                    except ValueError:
                        pass

    def close(self) -> None:
        """Emit the SLA histograms. Idempotent.

        Only successful requests are observed: a 5xx contributes no meaningful
        latency and would drag the planner's window averages toward zero.
        """
        if self._closed:
            return
        self._closed = True
        if self["outcome"] != "ok":
            return
        router = self._router

        osl = self._osl if self._osl is not None else self._frames
        if self._isl is not None:
            input_sequence_tokens.labels(router=router).observe(self._isl)
        if osl > 0:
            output_sequence_tokens.labels(router=router).observe(osl)

        if self._ttft is None:
            # Non-streaming reply: the whole response landed at once, so there
            # is no observable first-token boundary to report.
            return
        time_to_first_token_seconds.labels(router=router).observe(self._ttft)
        if osl > 1:
            decode_time = max(0.0, time.perf_counter() - self._start - self._ttft)
            inter_token_latency_seconds.labels(router=router).observe(decode_time / (osl - 1))


@contextmanager
def track_request(router: str):
    """Context manager that wraps a request's server-side lifetime to
    populate `request_duration_seconds` (with outcome) and the in-flight
    gauge. Use like::

        with track_request(router="mixed") as obs:
            resp = await dispatch(...)
            obs["outcome"] = "ok" if resp.status_code < 400 else f"{resp.status_code // 100}xx"

    Yields a :class:`RequestObserver`, whose extra methods feed the SLA
    histograms. For a streaming reply the observer is handed to the stream
    generator, which closes it once the last token has been forwarded.
    """
    obs = RequestObserver(router)
    request_inflight.labels(router=router).inc()
    start = time.perf_counter()
    try:
        yield obs
    finally:
        request_inflight.labels(router=router).dec()
        request_duration_seconds.labels(router=router, outcome=obs["outcome"]).observe(
            time.perf_counter() - start
        )
        if not obs.deferred:
            obs.close()


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
