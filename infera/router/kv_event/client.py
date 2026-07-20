###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""KvEventClient: per-worker ZMQ subscriber + cache view maintenance.

Plug ``on_worker_added`` / ``on_worker_removed`` into a Registry to
auto-track the fleet.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import zmq
import zmq.asyncio
from msgspec.msgpack import Decoder

from infera.common.worker_pool import WorkerInfo
from infera.router.kv_event.events import (
    ALL_CLEARED_TYPES,
    BLOCK_REMOVED_TYPES,
    BLOCK_STORED_TYPES,
    SglangKVEventBatch,
    batch_type_for_engine,
)
from infera.router.kv_event.hasher import ROUTER_SEED, hash_chunk
from infera.router.policy.target import is_rank_multiplexed

logger = logging.getLogger(__name__)

_TOPIC = b"kv-events"
_INITIAL_BACKOFF = 0.1
_MAX_BACKOFF = 5.0


def _offset_endpoint(endpoint: str, rank: int) -> str:
    """Offset an endpoint's port by ``rank``, mirroring SGLang's
    ``offset_endpoint_port``: under ``--dp-size N`` each DP rank publishes
    kv-events on ``base_port + rank``. ``rank == 0`` returns the base."""
    if rank == 0:
        return endpoint
    head, _, port = endpoint.rpartition(":")
    return f"{head}:{int(port) + rank}"


@dataclass
class WorkerSubscription:
    """Per-worker subscription state: one ZMQ task per DP rank plus a
    chained-hash view + translation map keyed by rank.

    A rank-multiplexed worker (SGLang ``--dp-size``) publishes each rank's
    events on a separate port (``base + rank``); we subscribe to all of
    them and keep a view/map per rank. Single-rank workers use rank 0.
    """

    worker_id: str
    endpoint: str
    block_size: int
    # Decoder for THIS worker's engine wire format (vLLM map vs SGLang array).
    # Defaults to the SGLang/array layout (the historical wire format);
    # on_worker_added always sets it explicitly from the worker's EngineType.
    decoder: Decoder = field(default_factory=lambda: Decoder(type=SglangKVEventBatch))
    multiplexed: bool = False
    views: dict[int, set[int]] = field(default_factory=dict)
    maps: dict[int, dict[int, int]] = field(default_factory=dict)  # worker_hash -> router_hash
    tasks: list[asyncio.Task] = field(default_factory=list)

    def view_for(self, rank: int | None) -> set[int]:
        return self.views.setdefault(rank or 0, set())

    def map_for(self, rank: int | None) -> dict[int, int]:
        return self.maps.setdefault(rank or 0, {})


class KvEventClient:
    """Maintains a router-side mirror of each worker's KV cache state."""

    def __init__(self) -> None:
        self._ctx = zmq.asyncio.Context()
        self._subs: dict[str, WorkerSubscription] = {}

    # --- public API ---

    def cache_view(self, worker_id: str, dp_rank: int | None = None) -> set[int]:
        """Return the chained-hash set of a worker's cached blocks for a DP
        rank (``dp_rank=None`` -> the worker's single/rank-0 view).

        The returned set is a reference to internal state — callers MUST
        treat it as read-only. (We don't copy to avoid O(N) per lookup;
        N can reach ~10^5.)
        """
        sub = self._subs.get(worker_id)
        if sub is None:
            return set()
        return sub.views.get(dp_rank if dp_rank is not None else 0, set())

    def on_worker_added(self, w: WorkerInfo) -> None:
        if not w.kv_events_endpoint or w.worker_id in self._subs:
            return
        sub = WorkerSubscription(
            worker_id=w.worker_id,
            endpoint=w.kv_events_endpoint,
            block_size=w.kv_block_size or 1,
            # vLLM and SGLang serialize kv-events differently (map vs array); pick
            # the decoder matching THIS worker's engine or every event fails to
            # decode and the view stays empty.
            decoder=Decoder(type=batch_type_for_engine(w.engine)),
            multiplexed=is_rank_multiplexed(w),
        )
        # One subscriber per DP rank. SGLang puts rank r's events on
        # base_port + r; a single-rank worker is just rank 0 on the base.
        ranks = range(w.dp_size) if sub.multiplexed else range(1)
        for r in ranks:
            endpoint = _offset_endpoint(w.kv_events_endpoint, r)
            sub.tasks.append(
                asyncio.create_task(self._run(sub, r, endpoint), name=f"kv-sub-{w.worker_id}-dp{r}")
            )
        self._subs[w.worker_id] = sub
        logger.info(
            "kv events: subscribing to %s (block_size=%d, ranks=%d) at %s",
            w.worker_id,
            sub.block_size,
            len(sub.tasks),
            w.kv_events_endpoint,
        )

    def on_worker_removed(self, worker_id: str) -> None:
        sub = self._subs.pop(worker_id, None)
        if sub is None:
            return
        for t in sub.tasks:
            t.cancel()
        total = sum(len(v) for v in sub.views.values())
        logger.info("kv events: unsubscribed %s (view had %d blocks)", worker_id, total)

    async def aclose(self) -> None:
        tasks = [t for s in self._subs.values() for t in s.tasks]
        for t in tasks:
            t.cancel()
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._subs.clear()
        self._ctx.term()

    # --- subscriber loop ---

    async def _run(self, sub: WorkerSubscription, rank: int, endpoint: str) -> None:
        """Outer loop: re-establish the subscriber socket on any failure."""
        backoff = _INITIAL_BACKOFF
        while True:
            try:
                await self._subscribe_once(sub, rank, endpoint)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "kv subscriber for %s dp%d errored (%s); retry in %.1fs",
                    sub.worker_id,
                    rank,
                    exc,
                    backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _MAX_BACKOFF)

    async def _subscribe_once(self, sub: WorkerSubscription, rank: int, endpoint: str) -> None:
        """Single connect+recv loop for one DP rank's endpoint. ``finally``
        guarantees the socket is closed on any exit path."""
        sock = self._ctx.socket(zmq.SUB)
        try:
            sock.connect(endpoint)
            sock.subscribe(_TOPIC)
            while True:
                frames = await sock.recv_multipart()
                try:
                    batch = sub.decoder.decode(frames[-1])
                except Exception as exc:
                    logger.warning("kv decode failed for %s: %s", sub.worker_id, exc)
                    continue
                for ev in batch.events:
                    self._handle_event(sub, ev, rank)
        finally:
            sock.close(linger=0)

    # --- event handlers ---

    def _handle_event(self, sub: WorkerSubscription, ev: object, rank: int | None = None) -> None:
        if isinstance(ev, BLOCK_STORED_TYPES):
            self._on_block_stored(sub, ev, rank)
        elif isinstance(ev, BLOCK_REMOVED_TYPES):
            view, m = sub.view_for(rank), sub.map_for(rank)
            for wh in ev.block_hashes:
                rh = m.pop(wh, None)
                if rh is not None:
                    view.discard(rh)
        elif isinstance(ev, ALL_CLEARED_TYPES):
            sub.view_for(rank).clear()
            sub.map_for(rank).clear()

    def _on_block_stored(self, sub: WorkerSubscription, ev: object, rank: int | None) -> None:
        view, m = sub.view_for(rank), sub.map_for(rank)
        if ev.parent_block_hash is None:
            parent = ROUTER_SEED
        else:
            parent = m.get(ev.parent_block_hash)
            if parent is None:
                # Chain broken: missing parent (cold-start window or out-of-order).
                # Dropping the event is the safe choice—worse case a future prefix
                # query misses by one block.
                return
        bs = sub.block_size
        n = len(ev.token_ids) // bs
        for i in range(n):
            chunk = ev.token_ids[i * bs : (i + 1) * bs]
            parent = hash_chunk(parent, chunk)
            view.add(parent)
            m[ev.block_hashes[i]] = parent
