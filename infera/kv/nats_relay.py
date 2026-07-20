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
from infera.router.kv_event.events import KVEventBatch

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
        self._decoder: Decoder[KVEventBatch] = Decoder(type=KVEventBatch)
        self._encoder = Encoder()
        self._kv = None
        self._last_write: dict[int, float] = {}
        self._dirty: dict[int, bool] = {}
        self._ctx: zmq.asyncio.Context | None = None
        self._sockets: list[zmq.asyncio.Socket] = []
        self._tasks: list[asyncio.Task] = []
        self._closing = False

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
            except Exception:
                continue
            for ev in batch.events:
                self._view_helper._handle_event(self._sub, ev, rank)
            self._dirty[rank] = True
            await self._maybe_write_bucket(rank)

    async def _maybe_write_bucket(self, rank: int) -> None:
        if self._kv is None or not self._dirty.get(rank):
            return
        now = time.monotonic()
        if now - self._last_write.get(rank, 0.0) < _BUCKET_WRITE_INTERVAL_S:
            return
        self._last_write[rank] = now
        self._dirty[rank] = False
        view = sorted(self._sub.view_for(rank))
        try:
            await self._kv.put(kv_key_for_worker(self._worker_id, rank), self._encoder.encode(view))
        except Exception as exc:
            logger.warning("KV relay bucket put failed (r%d): %s", rank, exc)

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
