###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Worker-side KV-event relay: engine ZMQ -> NATS (+ JetStream KV bucket).

In NATS mode the engine keeps publishing KV events on its native ZMQ socket;
this relay (running inside the worker) does two things per DP rank:

  1. **Live**: republishes each batch verbatim onto
     ``infera.kv.events.<wid>.<rank>`` for low-latency router updates.
  2. **Bootstrap/self-heal**: maintains the authoritative router-side cache
     view (reusing the router's chained-hash logic) and writes it, throttled,
     into a JetStream **KV bucket** keyed by ``(worker, rank)``. A cold/
     reconnecting router reads the bucket for an instant bootstrap + resync,
     replacing the old HTTP ``/v1/kv-snapshot`` + ``SnapshotReconciler`` pull.

Single-rank workers use rank 0. SGLang ``--dp-size N`` multiplexes ranks on
``base_port + r``; the relay tails each rank's port and keeps them separate.
"""

from __future__ import annotations

import asyncio
import logging
import time
from urllib.parse import urlparse

import zmq
import zmq.asyncio
from msgspec.msgpack import Decoder, Encoder

from infera.kv.nats_bus import (
    NatsBus,
    kv_key_for_worker,
    subject_for_worker,
)
from infera.router.kv_event.client import KvEventClient, WorkerSubscription, _offset_endpoint
from infera.router.kv_event.events import ALL_CLEARED_TYPES, batch_type_for_engine

logger = logging.getLogger(__name__)

_TOPIC = b"kv-events"
# Don't hammer the KV bucket: coalesce writes to at most one per interval.
_BUCKET_WRITE_INTERVAL_S = 2.0


def _local_endpoint(endpoint: str) -> str:
    parsed = urlparse(endpoint)
    port = parsed.port
    if port is None:
        return endpoint
    return f"tcp://127.0.0.1:{port}"


class KvEventNatsRelay:
    """Forwards one worker's engine KV-event stream(s) onto NATS and mirrors
    its per-rank cache view into the JetStream KV bucket."""

    def __init__(
        self,
        *,
        worker_id: str,
        engine_zmq_endpoint: str,
        engine=None,
        block_size: int = 1,
        dp_size: int | None = None,
        multiplexed: bool = False,
        nats_url: str | None = None,
    ) -> None:
        self._worker_id = worker_id
        self._base_endpoint = _local_endpoint(engine_zmq_endpoint)
        # Ranks this relay tails. Multiplexed SGLang DP publishes rank r on
        # base_port + r; everything else is just rank 0.
        self._ranks = list(range(dp_size or 1)) if (multiplexed and dp_size) else [0]
        self._bus = NatsBus(nats_url)
        self._view_helper = KvEventClient()  # lends _handle_event only
        self._sub = WorkerSubscription(
            worker_id=worker_id, endpoint="(relay)", block_size=block_size or 1
        )
        # The wire format is per-engine -- vLLM/ATOM emit tagged maps, SGLang
        # tagged arrays -- so the decoder has to follow the worker's engine. A
        # fixed one decodes nothing at all for the other family, and because the
        # live forward below happens on the raw bytes, that failure is invisible
        # from the router: events flow, only the bucket silently stays empty.
        self._decoder: Decoder = Decoder(type=batch_type_for_engine(engine))
        self._encoder = Encoder()
        self._kv = None
        self._last_write: dict[int, float] = {}
        self._dirty: dict[int, bool] = {}
        self._decode_failures = 0
        self._next_decode_warn = 1
        self._ctx: zmq.asyncio.Context | None = None
        self._sockets: list[zmq.asyncio.Socket] = []
        self._tasks: list[asyncio.Task] = []
        self._closing = False
        # Set once the relay has decoded an ``AllBlocksCleared``. The engine
        # binds its KV-event PUB at launch and warms up long before this relay
        # connects, so the single ``parent_block_hash=None`` anchor event is
        # already gone by the time anyone subscribes -- and every later event
        # chains off an anchor the router never saw. Flushing the engine's cache
        # re-emits that anchor; this event is how the flusher knows its flush
        # actually landed on a live subscription rather than into the same void.
        self.cleared_observed = asyncio.Event()

    async def start(self) -> None:
        await self._bus.connect()
        await self._bus.ensure_event_stream()
        self._kv = await self._bus.kv_view_store()
        self._ctx = zmq.asyncio.Context.instance()
        for rank in self._ranks:
            endpoint = _offset_endpoint(self._base_endpoint, rank)
            sock = self._ctx.socket(zmq.SUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.connect(endpoint)
            sock.subscribe(_TOPIC)
            self._sockets.append(sock)
            self._tasks.append(
                asyncio.create_task(
                    self._loop(rank, sock), name=f"kv-nats-relay-{self._worker_id}-r{rank}"
                )
            )
        self._tasks.append(
            asyncio.create_task(self._drain_dirty(), name=f"kv-nats-relay-{self._worker_id}-bucket")
        )
        logger.info(
            "KV NATS relay up: %s ranks=%s -> %s (kv_bucket=%s)",
            self._base_endpoint,
            self._ranks,
            self._bus.url,
            "on" if self._kv is not None else "off",
        )

    async def _loop(self, rank: int, sock: zmq.asyncio.Socket) -> None:
        subject = subject_for_worker(self._worker_id, rank)
        while not self._closing:
            try:
                frames = await sock.recv_multipart()
            except asyncio.CancelledError:
                return
            except zmq.ZMQError as exc:
                logger.warning("KV relay ZMQ recv failed (r%d): %s", rank, exc)
                await asyncio.sleep(0.05)
                continue
            if not frames:
                continue
            payload = frames[-1]
            # 1. Live forward (verbatim) onto the durable JetStream stream.
            try:
                await self._bus.js_publish(subject, payload)
            except Exception as exc:
                logger.warning("KV relay NATS publish failed: %s", exc)
            # 2. Update authoritative per-rank view + persist to KV bucket.
            try:
                batch = self._decoder.decode(payload)
            except Exception as exc:
                self._note_decode_failure(rank, exc)
                continue
            for ev in batch.events:
                self._view_helper._handle_event(self._sub, ev, rank)
                if isinstance(ev, ALL_CLEARED_TYPES):
                    self.cleared_observed.set()
            self._dirty[rank] = True
            await self._maybe_write_bucket(rank)

    async def _drain_dirty(self) -> None:
        """Write out ranks the coalescing interval passed over.

        ``_maybe_write_bucket`` is only ever reached from ``_loop``, i.e. by the
        *next* event on that rank. An event landing inside the interval leaves
        the rank dirty and returns, so the tail of every burst -- the part that
        matters most, being the newest -- stayed unwritten for as long as the
        rank was quiet. That is the mechanism behind a stale bucket: a cold
        router seeds from a view that stops at the last write before the burst,
        and the router's own chain-health accounting reads the lapse as the
        relay having died.

        Polling is the whole implementation on purpose. A per-rank timer would
        have to be created, cancelled and re-armed on the event path, which is
        the hot path here; one task ticking twice per interval costs a dict
        lookup per rank per tick and cannot leak a timer.
        """
        while not self._closing:
            await asyncio.sleep(_BUCKET_WRITE_INTERVAL_S / 2)
            for rank in self._ranks:
                if self._closing:
                    return
                try:
                    await self._maybe_write_bucket(rank)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - a tick must not end the loop
                    logger.warning("KV relay bucket drain failed (r%d): %s", rank, exc)

    def _note_decode_failure(self, rank: int, exc: Exception) -> None:
        """Report batches this relay cannot read, geometrically.

        A decode failure here does not stop the live forward -- that already
        happened, on the raw bytes -- so the only visible symptom is a KV bucket
        that never fills, which looks exactly like a worker that has served no
        traffic. Staying silent about it is what let a wrong decoder survive
        three releases, so the first failure says so and the rate then backs off
        rather than filling the log at event rate.
        """
        self._decode_failures += 1
        if self._decode_failures < self._next_decode_warn:
            return
        logger.warning(
            "KV relay cannot decode the engine's events (r%d, %d so far): %s: %s -- "
            "live forwarding still works, but the KV bucket bootstrap is dead, so a "
            "router that starts cold has nothing to resync from",
            rank,
            self._decode_failures,
            type(exc).__name__,
            exc,
        )
        self._next_decode_warn = self._decode_failures * 10

    async def _maybe_write_bucket(self, rank: int) -> None:
        if self._kv is None or not self._dirty.get(rank):
            return
        now = time.monotonic()
        if now - self._last_write.get(rank, 0.0) < _BUCKET_WRITE_INTERVAL_S:
            return
        self._last_write[rank] = now
        # Cleared before the await, not after: an event that lands while the put
        # is in flight describes a view this put does not carry, and must leave
        # the rank dirty so the next tick writes it.
        self._dirty[rank] = False
        view = sorted(self._sub.view_for(rank))
        try:
            await self._kv.put(kv_key_for_worker(self._worker_id, rank), self._encoder.encode(view))
        except Exception as exc:
            # Nothing was written, so the bucket still holds an older view. Left
            # clean, that view would stand until the *next* event happened to
            # dirty the rank again -- and on a rank that has gone quiet, forever.
            # A stale bucket does not read to the router as stale: it reads as
            # coverage, and once it ages past SEED_COVERAGE_WINDOW, as a relay
            # that died -- which is what arms the destructive self-heal flush.
            # `_last_write` stays stamped, so the retry is spaced like any other
            # write rather than spinning at drain-tick rate against a dead bus.
            self._dirty[rank] = True
            logger.warning("KV relay bucket put failed (r%d), will retry: %s", rank, exc)

    async def stop(self) -> None:
        self._closing = True
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        self._tasks = []
        for sock in self._sockets:
            sock.close(linger=0)
        self._sockets = []
        # Best-effort final flush so a clean shutdown leaves fresh views.
        if self._kv is not None:
            for rank in self._ranks:
                try:
                    await self._kv.put(
                        kv_key_for_worker(self._worker_id, rank),
                        self._encoder.encode(sorted(self._sub.view_for(rank))),
                    )
                except Exception:
                    pass
        await self._bus.close()
