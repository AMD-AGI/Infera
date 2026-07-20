###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""ZMQ PUB-side of the KV event transport. One per worker / pool daemon.

Bind once at the configured endpoint. Probes (or the daemon's own event
emitter) call `emit()` synchronously to enqueue events; a background
batcher coalesces events by (model_name, compat_key) and sends them as
msgpack-framed multipart ZMQ messages.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import zmq
import zmq.asyncio

from infera.kv.wire import (
    EventBatch,
    NormalizedEvent,
    encode_batch,
    encode_topic,
)

logger = logging.getLogger(__name__)

# ZMQ socket high-water-mark: messages above this get dropped at the
# socket level when the subscriber falls behind. Snapshot reconcile
# recovers any dropped state.
DEFAULT_SNDHWM = 16384


@dataclass
class _StreamState:
    """Per-(model, compat_key) batching state."""

    buffer: list[NormalizedEvent] = field(default_factory=list)
    next_batch_id: int = 0


class KvEventPublisher:
    """Worker / daemon-side ZMQ PUB for KV cache events.

    Lifecycle:
        pub = KvEventPublisher(
            bind_endpoint="tcp://0.0.0.0:5557",
            publisher_id="10.0.0.5:30000",
            publisher_type="worker",
            index_block_size=64,
        )
        await pub.start()
        pub.emit(model="Qwen3.6", compat_key="abc", event=normalized_event)
        ...
        await pub.stop()

    Concurrency: `emit` is sync and may be called from any task (it appends
    to per-stream buffers; protected by a small lock). The batcher task
    is the sole consumer of those buffers.
    """

    def __init__(
        self,
        *,
        bind_endpoint: str,
        publisher_id: str,
        publisher_type: str,
        index_block_size: int,
        batch_max_events: int = 256,
        batch_max_ms: int = 100,
        sndhwm: int = DEFAULT_SNDHWM,
        slow_joiner_pause_ms: int = 100,
    ) -> None:
        self._bind_endpoint = bind_endpoint
        self._publisher_id = publisher_id
        self._publisher_type = publisher_type
        self._index_block_size = index_block_size
        self._batch_max_events = batch_max_events
        self._batch_max_ms = batch_max_ms
        self._sndhwm = sndhwm
        self._slow_joiner_pause_ms = slow_joiner_pause_ms

        self._ctx: zmq.asyncio.Context | None = None
        self._socket: zmq.asyncio.Socket | None = None
        self._streams: dict[tuple[str, str], _StreamState] = {}
        self._buf_lock = asyncio.Lock()
        self._batcher_task: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._started = False
        self._closing = False

    async def start(self) -> None:
        if self._started:
            return
        self._ctx = zmq.asyncio.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, self._sndhwm)
        # LINGER=0 means close() drops unsent messages immediately; we
        # explicitly flush in stop() instead.
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(self._bind_endpoint)
        # ZMQ slow-joiner: after bind, give subscribers a moment to
        # connect before the first send. Standard ZMQ idiom.
        await asyncio.sleep(self._slow_joiner_pause_ms / 1000)
        self._batcher_task = asyncio.create_task(self._batcher_loop(), name="kv-publisher-batcher")
        self._started = True
        logger.info(
            "KvEventPublisher started: endpoint=%s publisher_id=%s",
            self._bind_endpoint,
            self._publisher_id,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._closing = True
        self._wake.set()
        if self._batcher_task is not None:
            try:
                await asyncio.wait_for(self._batcher_task, timeout=1.0)
            except asyncio.TimeoutError:
                self._batcher_task.cancel()
                try:
                    await self._batcher_task
                except (asyncio.CancelledError, Exception):
                    pass
        # Flush whatever is left.
        await self._flush_all()
        if self._socket is not None:
            self._socket.close(linger=0)
            self._socket = None
        self._started = False
        logger.info("KvEventPublisher stopped: publisher_id=%s", self._publisher_id)

    def emit(self, *, model: str, compat_key: str, event: NormalizedEvent) -> None:
        """Enqueue one event. Sync (no await). Triggers an immediate flush
        if the buffer for that (model, compat_key) reaches `batch_max_events`.
        """
        if not self._started or self._closing:
            return
        key = (model, compat_key)
        state = self._streams.get(key)
        if state is None:
            state = _StreamState()
            self._streams[key] = state
        state.buffer.append(event)
        if len(state.buffer) >= self._batch_max_events:
            self._wake.set()

    async def _batcher_loop(self) -> None:
        """Wake on timer OR on `_wake` event (set by emit when a buffer fills)."""
        try:
            while not self._closing:
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        timeout=self._batch_max_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    pass
                self._wake.clear()
                await self._flush_all()
        except asyncio.CancelledError:
            return

    async def _flush_all(self) -> None:
        if self._socket is None:
            return
        # Snapshot streams to avoid holding state while sending. Each stream's
        # buffer is swapped out under the lock; emit appends to the new buffer.
        async with self._buf_lock:
            to_send: list[tuple[tuple[str, str], list[NormalizedEvent], int]] = []
            for key, state in self._streams.items():
                # Chunk the buffer into batches of <= batch_max_events. This
                # ensures a fast burst of emits at small batch_max_events
                # produces multiple batches (each with its own batch_id),
                # rather than coalescing into one over-large batch.
                while state.buffer:
                    chunk = state.buffer[: self._batch_max_events]
                    state.buffer = state.buffer[self._batch_max_events :]
                    batch_id = state.next_batch_id
                    state.next_batch_id += 1
                    to_send.append((key, chunk, batch_id))
        for (model, compat_key), events, batch_id in to_send:
            batch = EventBatch(
                publisher_id=self._publisher_id,
                publisher_type=self._publisher_type,
                model_name=model,
                compat_key=compat_key,
                index_block_size=self._index_block_size,
                batch_id=batch_id,
                events=tuple(events),
            )
            payload = encode_batch(batch)
            try:
                await self._socket.send_multipart([encode_topic(model), payload])
            except zmq.ZMQError as exc:
                logger.warning("ZMQ send_multipart failed: %s", exc)
