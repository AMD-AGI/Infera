###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""HTTP surfaces for KV management.

Three pieces, all sharing one module so the wiring is in one place:

  - `make_snapshot_router(producer)` — worker-side FastAPI router that
    exposes `GET /v1/kv-snapshot?model=X&compat_key=Y`. The engine's
    main HTTP app mounts this.

  - `make_stats_router(index=..., writer=..., subscribers=..., reconciler=...)`
    — server-side FastAPI router exposing `GET /v1/kv-stats`. Returns
    a flat JSON dict with counters from each component; suitable for
    Prometheus scrape via a thin exporter or for ad-hoc curl debugging.

  - `make_http_snapshot_puller(client)` — server-side. Returns a
    `PullFn` compatible with `SnapshotReconciler(pull_fn=...)`, doing
    HTTP GET against a worker's `/v1/kv-snapshot`.

The two server-side pieces are kept minimal: counters out, no
mutation. Operator-triggered actions (e.g. invalidate) belong in a
separate admin surface (`19-trust-and-deployment.md`).
"""

from __future__ import annotations

import logging
from dataclasses import asdict

import httpx
from fastapi import APIRouter, Query

from infera.kv.index import KVIndex
from infera.kv.snapshot import SnapshotProducer, SnapshotReconciler
from infera.kv.subscriber import KvEventSubscriberPool
from infera.kv.wire import Snapshot, snapshot_from_json, snapshot_to_json
from infera.kv.writer import KvIndexWriter

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Worker side: /v1/kv-snapshot
# ----------------------------------------------------------------------


def make_snapshot_router(producer: SnapshotProducer) -> APIRouter:
    """Build a FastAPI router that serves the publisher's snapshot.

    The engine adapter mounts this on its existing FastAPI app so the
    snapshot endpoint sits alongside the engine's own endpoints
    (`/v1/chat/completions`, `/health`, etc.).
    """
    router = APIRouter()

    @router.get("/v1/kv-snapshot")
    async def get_snapshot(
        model: str = Query(..., description="Model name to snapshot"),
        compat_key: str = Query(..., description="Compat key namespace"),
    ) -> dict:
        # The producer always returns *something* — even if the publisher
        # has never seen events for this (model, compat_key), it returns
        # an empty snapshot with batch_id=-1. The reconciler handles that.
        snapshot = producer.snapshot(model=model, compat_key=compat_key)
        return snapshot_to_json(snapshot)

    return router


# ----------------------------------------------------------------------
# Server side: /v1/kv-stats
# ----------------------------------------------------------------------


def make_stats_router(
    *,
    index: KVIndex,
    writer: KvIndexWriter | None = None,
    subscribers: KvEventSubscriberPool | None = None,
    reconciler: SnapshotReconciler | None = None,
) -> APIRouter:
    """Build a FastAPI router that exposes per-component KV counters.

    Any subset of writer/subscribers/reconciler may be None — the
    response just omits that section.
    """
    router = APIRouter()

    @router.get("/v1/kv-stats")
    async def get_stats() -> dict:
        out: dict = {
            "index": asdict(index.stats()),
        }
        if writer is not None:
            m = writer.metrics
            out["writer"] = {
                "batches_applied": m.batches_applied,
                "events_applied_stored": m.events_applied_stored,
                "events_applied_removed": m.events_applied_removed,
                "events_applied_cleared": m.events_applied_cleared,
                "events_dropped_unknown_role": m.events_dropped_unknown_role,
                "events_dropped_unknown_tier": m.events_dropped_unknown_tier,
                "events_dropped_malformed": m.events_dropped_malformed,
            }
        if subscribers is not None:
            m = subscribers.aggregate_metrics()
            out["subscribers"] = {
                "endpoints": subscribers.endpoints(),
                "batches_received": m.batches_received,
                "batches_decoded": m.batches_decoded,
                "events_received": m.events_received,
                "batches_dropped_overload": m.batches_dropped_overload,
                "batches_dropped_malformed": m.batches_dropped_malformed,
                "batches_dropped_version": m.batches_dropped_version,
            }
        if reconciler is not None:
            out["reconciler"] = {
                "snapshots_pulled": reconciler.snapshots_pulled,
                "snapshots_applied": reconciler.snapshots_applied,
                "snapshots_failed": reconciler.snapshots_failed,
                "snapshots_stale": reconciler.snapshots_stale,
            }
        return out

    return router


# ----------------------------------------------------------------------
# Server side: HTTP snapshot puller
# ----------------------------------------------------------------------


def make_http_snapshot_puller(
    *,
    client: httpx.AsyncClient | None = None,
    timeout: float = 10.0,
    path: str = "/v1/kv-snapshot",
):
    """Build a `pull_fn` for `SnapshotReconciler` that does HTTP GET.

    `endpoint` passed in the call is treated as the publisher's HTTP
    base URL (e.g., `http://10.0.0.5:30000`). The function appends
    `path` and includes `model` + `compat_key` as query parameters.

    Returns None on transient failure so the reconciler's retry logic
    handles it. Raises only on truly programmer errors.
    """
    # If no client is given, create one per-call (less efficient but
    # works without lifecycle management; tests pass a shared client).
    shared_client = client

    async def pull(
        publisher_id: str,
        endpoint: str,
        model: str,
        compat_key: str,
    ) -> Snapshot | None:
        c = shared_client or httpx.AsyncClient(timeout=timeout)
        try:
            url = endpoint.rstrip("/") + path
            try:
                resp = await c.get(
                    url,
                    params={"model": model, "compat_key": compat_key},
                )
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                logger.warning(
                    "snapshot HTTP GET failed for %s @ %s (%s/%s): %s",
                    publisher_id,
                    endpoint,
                    model,
                    compat_key,
                    exc,
                )
                return None
            if resp.status_code != 200:
                logger.warning(
                    "snapshot HTTP GET returned %d for %s @ %s (%s/%s)",
                    resp.status_code,
                    publisher_id,
                    endpoint,
                    model,
                    compat_key,
                )
                return None
            try:
                obj = resp.json()
            except ValueError as exc:
                logger.warning("snapshot response not JSON from %s: %s", endpoint, exc)
                return None
            try:
                return snapshot_from_json(obj)
            except ValueError as exc:
                logger.warning(
                    "snapshot decode failed from %s (%s/%s): %s",
                    endpoint,
                    model,
                    compat_key,
                    exc,
                )
                return None
        finally:
            if shared_client is None:
                await c.aclose()

    return pull
