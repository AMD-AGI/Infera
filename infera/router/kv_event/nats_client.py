###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""NATS-fed variant of :class:`KvEventClient`.

Same router-side cache-view bookkeeping as the ZMQ client (it subclasses it
and reuses ``_handle_event`` / ``cache_view`` / ``WorkerSubscription``), but
the transport is a single NATS subscription to ``infera.kv.events.>``
instead of one ZMQ SUB socket per worker. Each message carries one worker's
``KVEventBatch`` (forwarded by that worker's :class:`KvEventNatsRelay`); the
worker_id is recovered from the subject.

Limitation: single-rank workers only (rank 0). DP-rank multiplexing
(``--dp-size``) would need the relay to tail each rank's port; not wired yet.
"""

from __future__ import annotations

import asyncio
import logging

from msgspec.msgpack import Decoder

from infera.common.worker_pool import WorkerInfo
from infera.kv.nats_bus import (
    KV_EVENTS_SUBJECT_PREFIX,
    NatsBus,
    parse_kv_key,
    parse_kv_subject,
)
from infera.router.kv_event.client import KvEventClient, WorkerSubscription
from infera.router.policy.target import is_rank_multiplexed

logger = logging.getLogger(__name__)


class NatsKvEventClient(KvEventClient):
    """KvEventClient whose events arrive over a NATS broker."""

    def __init__(self, nats_url: str | None = None) -> None:
        super().__init__()
        self._bus = NatsBus(nats_url)
        self._sub = None
        self._started = False
        self._start_lock = asyncio.Lock()
        self._kv = None
        self._view_decoder: Decoder[list] = Decoder(type=list)
        self._resync_task: asyncio.Task | None = None

    def on_worker_added(self, w: WorkerInfo) -> None:
        # No per-worker socket in NATS mode: just record the subscription
        # state (block_size, multiplexed) so cache_view / _handle_event have
        # somewhere to write. Ingestion is the single global NATS sub.
        if w.worker_id in self._subs:
            return
        self._subs[w.worker_id] = WorkerSubscription(
            worker_id=w.worker_id,
            endpoint="(nats)",
            block_size=w.kv_block_size or 1,
            multiplexed=is_rank_multiplexed(w),
        )
        logger.info(
            "kv events (nats): tracking %s (block_size=%d)",
            w.worker_id,
            w.kv_block_size or 1,
        )
        # Lazily bring up the broker subscription on first worker.
        asyncio.create_task(self._ensure_started(), name="kv-nats-start")

    def on_worker_removed(self, worker_id: str) -> None:
        sub = self._subs.pop(worker_id, None)
        if sub is not None:
            total = sum(len(v) for v in sub.views.values())
            logger.info("kv events (nats): untracked %s (view had %d blocks)", worker_id, total)

    async def _ensure_started(self) -> None:
        async with self._start_lock:
            if self._started:
                return
            await self._bus.connect()
            # Live incremental deltas over a durable JetStream stream (no
            # silent drop on slow consumers; ephemeral consumer => fan-out).
            await self._bus.ensure_event_stream()
            self._sub = await self._bus.js_subscribe(
                f"{KV_EVENTS_SUBJECT_PREFIX}.>", self._on_message
            )
            # JetStream KV bucket bootstrap + self-heal via a SINGLE watchall
            # consumer (push): delivers all current keys first (cold-start
            # bootstrap), then ongoing PUTs (drift resync as the relay rewrites
            # views). One consumer total -- avoids the per-poll consumer churn a
            # keys()/get() loop would create on the bucket stream.
            self._kv = await self._bus.kv_view_store()
            self._resync_task = asyncio.create_task(self._watch_bucket(), name="kv-nats-watch")
            self._started = True
            logger.info(
                "kv events (nats): subscribed %s.> + KV-bucket watchall bootstrap/resync",
                KV_EVENTS_SUBJECT_PREFIX,
            )

    async def _watch_bucket(self) -> None:
        """Watch the KV bucket: initial values bootstrap each worker's view,
        subsequent PUTs self-heal drift. Replaces the HTTP snapshot reconciler
        with a single push consumer."""
        if self._kv is None:
            return
        try:
            watcher = await self._kv.watchall()
        except Exception:
            logger.exception("kv events (nats): watchall failed")
            return
        try:
            async for entry in watcher:
                if entry is None:
                    continue  # nats-py end-of-initial-values marker
                parsed = parse_kv_key(entry.key)
                if parsed is None:
                    continue
                worker_id, rank = parsed
                sub = self._subs.get(worker_id)
                if sub is None:
                    continue  # not tracking this worker (yet)
                op = getattr(entry, "operation", None)
                if op in ("DEL", "DELETE", "PURGE"):
                    sub.views.pop(rank, None)
                    continue
                try:
                    snapshot = set(self._view_decoder.decode(entry.value))
                except Exception:
                    continue
                # The durable-stream replay (DeliverPolicy.ALL via _on_message)
                # is the authoritative, ordered view builder. The bucket is only
                # a cold-start bootstrap shortcut, so: (a) never apply an empty
                # snapshot (a relay that desynced can publish an empty view and
                # would otherwise clobber a good incremental view -> cache_hits
                # collapse to 0), and (b) only seed when we don't already have an
                # incremental view for this rank.
                if not snapshot:
                    continue
                if not sub.views.get(rank):
                    sub.views[rank] = snapshot
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("kv events (nats): watch loop error")

    async def _on_message(self, subject: str, data: bytes) -> None:
        parsed = parse_kv_subject(subject)
        if parsed is None:
            return
        worker_id, rank = parsed
        sub = self._subs.get(worker_id)
        if sub is None:
            # Event arrived before the worker registered (or after removal);
            # safe to drop — a snapshot/late registration will reconcile.
            return
        try:
            batch = self._decoder.decode(data)
        except Exception as exc:
            logger.warning("kv events (nats): decode failed for %s: %s", worker_id, exc)
            return
        for ev in batch.events:
            self._handle_event(sub, ev, rank)

    async def aclose(self) -> None:
        if self._resync_task is not None:
            self._resync_task.cancel()
            try:
                await self._resync_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._bus.close()
        await super().aclose()
