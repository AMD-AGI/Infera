###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ZMQ SUB-side of the KV event transport. One per (server replica × publisher endpoint).

Each server replica runs a fleet of subscribers — one per known
publisher endpoint discovered via etcd. Each subscriber drains its
ZMQ SUB socket into a shared `asyncio.Queue[EventBatch]` consumed by
the single-writer KvIndexWriter task (see `writer.py`).

Backpressure: if the queue is full, the subscriber drops the batch
and increments `kv_event_queue_dropped_total`. The snapshot reconciler
catches drift; ZMQ PUB at the publisher already drops on slow consumers
via `ZMQ_RCVHWM`, so loss is expected and recoverable.
"""

from __future__ import annotations

import asyncio
import logging

import zmq
import zmq.asyncio

from infera.kv.wire import decode_batch

logger = logging.getLogger(__name__)

DEFAULT_RCVHWM = 16384


class SubscriberMetrics:
    """Per-subscriber observability counters. Single-threaded read/write
    so we don't need locks; metrics scraper reads the int snapshots."""

    def __init__(self) -> None:
        self.batches_received = 0
        self.batches_decoded = 0
        self.events_received = 0
        self.batches_dropped_overload = 0
        self.batches_dropped_malformed = 0
        self.batches_dropped_version = 0


class KvEventSubscriber:
    """One SUB socket connected to one publisher endpoint."""

    def __init__(
        self,
        *,
        endpoint: str,
        output_queue: asyncio.Queue,
        topic_filter: str | None = None,
        rcvhwm: int = DEFAULT_RCVHWM,
    ) -> None:
        self._endpoint = endpoint
        self._output_queue = output_queue
        # If set, only subscribe to messages whose first frame matches.
        # None → subscribe to all topics (empty-byte filter).
        self._topic_filter = topic_filter
        self._rcvhwm = rcvhwm

        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._task: asyncio.Task | None = None
        self._started = False
        self._closing = False
        self.metrics = SubscriberMetrics()

    async def start(self) -> None:
        if self._started:
            return
        self._ctx = zmq.asyncio.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, self._rcvhwm)
        self._socket.setsockopt(zmq.LINGER, 0)
        if self._topic_filter is None:
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        else:
            self._socket.setsockopt(zmq.SUBSCRIBE, self._topic_filter.encode("utf-8"))
        self._socket.connect(self._endpoint)
        self._task = asyncio.create_task(self._recv_loop(), name=f"kv-sub-{self._endpoint}")
        self._started = True
        logger.info("KvEventSubscriber connected: endpoint=%s", self._endpoint)

    async def stop(self) -> None:
        if not self._started:
            return
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._started = False
        logger.info("KvEventSubscriber disconnected: endpoint=%s", self._endpoint)

    async def _recv_loop(self) -> None:
        assert self._socket is not None
        while not self._closing:
            try:
                parts = await self._socket.recv_multipart()
            except asyncio.CancelledError:
                return
            except zmq.ZMQError as exc:
                logger.warning("ZMQ recv failed on %s: %s", self._endpoint, exc)
                await asyncio.sleep(0.05)
                continue

            if len(parts) != 2:
                self.metrics.batches_dropped_malformed += 1
                continue
            _topic, payload = parts
            self.metrics.batches_received += 1
            try:
                batch = decode_batch(payload)
            except Exception as exc:
                # PR #9 review fix P1 (subscriber error scope):
                # decode_batch can raise ValueError (event_version /
                # malformed msgpack), but also KeyError (missing wire
                # field) or msgpack.UnpackException (binary garbage).
                # Earlier code caught only ValueError and a non-
                # ValueError raise here would kill `_recv_loop`,
                # leaving the subscriber "started" but receiving
                # nothing — silent degradation. Catch broadly and
                # keep the loop alive.
                if isinstance(exc, ValueError) and (
                    "event_version" in str(exc) or "newer than supported" in str(exc)
                ):
                    self.metrics.batches_dropped_version += 1
                else:
                    self.metrics.batches_dropped_malformed += 1
                logger.warning(
                    "malformed batch from %s (%s): %s",
                    self._endpoint,
                    type(exc).__name__,
                    exc,
                )
                continue

            self.metrics.batches_decoded += 1
            self.metrics.events_received += len(batch.events)

            try:
                self._output_queue.put_nowait(batch)
            except asyncio.QueueFull:
                self.metrics.batches_dropped_overload += 1
                logger.warning(
                    "writer queue full; dropping batch from %s (publisher_id=%s, batch_id=%d)",
                    self._endpoint,
                    batch.publisher_id,
                    batch.batch_id,
                )


class KvEventSubscriberPool:
    """Manages a set of subscribers, one per publisher endpoint.

    Server-side caller adds endpoints discovered via etcd; pool spawns
    a subscriber per endpoint. On endpoint removal, the subscriber is
    stopped and dropped. All batches feed one shared queue.
    """

    def __init__(self, output_queue: asyncio.Queue) -> None:
        self._output_queue = output_queue
        self._subscribers: dict[str, KvEventSubscriber] = {}

    async def add(self, endpoint: str) -> None:
        if endpoint in self._subscribers:
            return
        sub = KvEventSubscriber(endpoint=endpoint, output_queue=self._output_queue)
        await sub.start()
        self._subscribers[endpoint] = sub

    async def remove(self, endpoint: str) -> None:
        sub = self._subscribers.pop(endpoint, None)
        if sub is not None:
            await sub.stop()

    async def stop_all(self) -> None:
        # Stop in parallel.
        await asyncio.gather(
            *(s.stop() for s in self._subscribers.values()),
            return_exceptions=True,
        )
        self._subscribers.clear()

    def endpoints(self) -> list[str]:
        return list(self._subscribers.keys())

    def aggregate_metrics(self) -> SubscriberMetrics:
        agg = SubscriberMetrics()
        for s in self._subscribers.values():
            agg.batches_received += s.metrics.batches_received
            agg.batches_decoded += s.metrics.batches_decoded
            agg.events_received += s.metrics.events_received
            agg.batches_dropped_overload += s.metrics.batches_dropped_overload
            agg.batches_dropped_malformed += s.metrics.batches_dropped_malformed
            agg.batches_dropped_version += s.metrics.batches_dropped_version
        return agg
