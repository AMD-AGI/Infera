###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""infera-kvd asyncio daemon.

Listens on a Unix domain socket. One connection per engine adapter
(SGLang or vLLM); the daemon multiplexes requests across connections
under a single `HostStore`.

Run::

    python -m infera.kvd --socket /var/run/infera-kvd.sock --max-bytes 32G

For tests, prefer an ephemeral socket path under `tmp_path`. The
daemon binds with `umask` so only the owning UID can connect — same
as the K8s pod-shared-filesystem pattern.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import threading
import uuid
from pathlib import Path

from infera.kvd.fd_passing import send_fd_async
from infera.kvd.store import HostStore
from infera.kvd.wire import (
    _VALID_RETENTIONS,
    BatchGet,
    BatchGetResponse,
    BatchGetSharedResponse,
    BatchSet,
    BatchSetAck,
    Clear,
    ClearAck,
    ErrorMessage,
    Exists,
    ExistsResponse,
    Get,
    GetResponse,
    GetSharedResponse,
    Hello,
    HelloAck,
    PrefetchHint,
    Set,
    SetAck,
    SetCancel,
    SetCancelResponse,
    SetCommit,
    SetCommitResponse,
    SetReserve,
    SetReserveResponse,
    Stats,
    StatsResponse,
    read_frame,
    write_frame,
)

logger = logging.getLogger(__name__)

_PROTOCOL_VERSION = 1

# Issue #20 item 3 — bounded set of recently warmed keys we use to
# detect "useful" prefetch hits. We don't need a per-key timestamp
# (TTL is enforced inside the store via `expires_at`); we only need
# to recognize the first `get` after the warm and count it.
_PREFETCH_WARMED_CAP = 4096


class KvdServer:
    """asyncio UDS server. Owns one `HostStore`. Spawns one connection
    handler per incoming client."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        max_bytes: int | None = None,
        server_id: str | None = None,
        store: HostStore | None = None,
        prefetch_inflight: int = 64,
    ) -> None:
        """`store` lets callers pre-construct a `HostStore` with SSD
        regions already wired. If not provided, builds a RAM-only store
        from `max_bytes` (Phase 3.0 backward-compatible).

        ``prefetch_inflight`` caps the speculative-prefetch worker's
        queue (issue #20 item 3 / PD §6.2). Hints exceeding the cap
        are dropped with a counter bump; the router can rate-limit
        upstream to avoid this. Default 64 = ~one full request's
        worth of blocks at typical 50K-token prompts on MiniMax-M2.5
        (32-tokens/block × 64 blocks ≈ 2K tokens of prefix; tune up
        for longer prefixes via `--prefetch-inflight`)."""
        self._socket_path = Path(socket_path)
        if store is not None:
            self._store = store
        elif max_bytes is not None:
            self._store = HostStore(max_bytes=max_bytes)
        else:
            raise ValueError("either `store` or `max_bytes` must be provided")
        self._server_id = server_id or f"kvd-{uuid.uuid4().hex[:8]}"
        self._server: asyncio.Server | None = None
        self._shutdown = asyncio.Event()

        # Issue #20 item 3 — speculative L3 prefetch worker.
        # See PD design §6.2 + §9 Phase 3 for the motivation. The
        # worker is the consumer side of PrefetchHint UDS frames:
        # it pops `(key, deadline_ms)` pairs from a bounded asyncio
        # queue, checks if the key is already in host RAM (cheap),
        # falls through to long region / spillover on miss, and
        # promotes the bytes into the RAM tier with the deadline as
        # a TTL so an un-asked-for warm-up doesn't pin RAM forever.
        # Lazy-spawned on `start()` so unit tests of `KvdServer`'s
        # data path don't have to manage a background task.
        self._prefetch_inflight = prefetch_inflight
        self._prefetch_queue: asyncio.Queue[tuple[bytes, str, str, int]] | None = None
        self._prefetch_task: asyncio.Task[None] | None = None
        # Stats for prefetch: hints received, fetches actually issued
        # (after the in-RAM dedupe filter), and "useful" fetches —
        # incremented when a `get` later hits a key the prefetch
        # worker just warmed. The last counter is what justifies the
        # whole mechanism in §9 Phase 3's exit criterion.
        self._prefetch_hints_total = 0
        self._prefetch_fetches_total = 0
        self._prefetch_dropped_total = 0
        # Keys recently warmed by the prefetch worker. Bounded set —
        # if it grows past `_PREFETCH_WARMED_CAP` we drop oldest.
        # Touched by `get` to count "useful" prefetches without
        # paying for a full per-entry timestamp.
        self._prefetch_warmed_keys: set[bytes] = set()
        self._prefetch_warmed_order: list[bytes] = []
        self._prefetch_hits_useful_total = 0

        # Per-connection ID for save-side CopyFree (zero-copy set)
        # ownership checks. The arena's reservation table keys leases
        # by this id so a misbehaving client can't commit/cancel
        # another connection's leases. Monotonic across the daemon's
        # lifetime; values are never reused. Lock is needed because
        # multiple `start_serving` connections can arrive in parallel.
        self._next_connection_id = 1
        self._connection_id_lock = threading.Lock()

    @property
    def store(self) -> HostStore:
        """Exposed for tests + future inspection endpoints."""
        return self._store

    @property
    def server_id(self) -> str:
        return self._server_id

    async def start(self) -> None:
        """Bind the socket, register a SIGTERM handler, return when ready.

        Caller awaits `serve_forever()` separately so it can do other
        setup (e.g. wire stats into Prometheus).
        """
        # Clean stale socket file from a previous crash.
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError:
                pass

        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        # umask 0o077 → 0700 socket perms (only owner can connect).
        old_umask = os.umask(0o077)
        try:
            self._server = await asyncio.start_unix_server(
                self._handle_client, path=str(self._socket_path)
            )
        finally:
            os.umask(old_umask)
        logger.info(
            "infera-kvd %s listening on %s (max=%d bytes)",
            self._server_id,
            self._socket_path,
            self._store.max_bytes,
        )
        # Spawn the prefetch worker now that the event loop is running.
        # We do this here rather than in __init__ because asyncio.Queue
        # binds to the running loop at construction.
        self._prefetch_queue = asyncio.Queue(maxsize=self._prefetch_inflight)
        self._prefetch_task = asyncio.create_task(
            self._run_prefetch_worker(), name="kvd-prefetch-worker"
        )

    async def serve_forever(self) -> None:
        """Block until `shutdown()` or SIGTERM. Use as the main blocking
        call of the daemon process."""
        if self._server is None:
            raise RuntimeError("call start() first")
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._shutdown.set)
            except NotImplementedError:
                # Windows / pytest doesn't always allow signal handlers.
                pass
        # asyncio.Server.wait_closed() doesn't return on signal; gate
        # on our own event instead and gracefully close when set.
        wait_serve = asyncio.create_task(self._server.serve_forever())
        wait_shutdown = asyncio.create_task(self._shutdown.wait())
        try:
            await asyncio.wait([wait_serve, wait_shutdown], return_when=asyncio.FIRST_COMPLETED)
        finally:
            wait_serve.cancel()
            wait_shutdown.cancel()
            await self.stop()

    async def stop(self) -> None:
        # Cancel the prefetch worker before tearing down the socket
        # — otherwise the worker can outlive the daemon by a few
        # cycles and produce warning logs after process intent-to-exit.
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except (asyncio.CancelledError, Exception):
                pass
            self._prefetch_task = None
            self._prefetch_queue = None
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        if self._socket_path.exists():
            try:
                self._socket_path.unlink()
            except OSError:
                pass
        logger.info("infera-kvd %s stopped", self._server_id)

    async def _run_prefetch_worker(self) -> None:
        """Background task consuming hints from the prefetch queue.

        Per hint key:
          1. If already in host RAM → skip (idempotent hint, common case
             when the router over-emits). Cheap dict lookup.
          2. Else fetch from long region, fall back to spillover.
             Both are async-friendly via `get_bytes`.
          3. On hit, write into host RAM via `store.set` with
             retention='short' + ttl_seconds=deadline_ms/1000. The TTL
             ensures a hint nobody follows through on doesn't pin RAM
             indefinitely (caller deadline is the natural bound).

        The worker is bounded by the queue capacity (`prefetch_inflight`)
        from `__init__`. Drops on overflow are counted by the dispatch
        handler — the worker itself never blocks on enqueueing.
        Cancellation via `stop()` is the normal shutdown signal.
        """
        assert self._prefetch_queue is not None
        while True:
            try:
                key, model, compat_key, deadline_ms = await self._prefetch_queue.get()
            except asyncio.CancelledError:
                return

            try:
                self._do_prefetch_one(key, model, compat_key, deadline_ms)
            except Exception:
                logger.exception(
                    "prefetch worker: failed to warm key=%s — skipping",
                    key.hex()[:16],
                )

    def _do_prefetch_one(self, key: bytes, model: str, compat_key: str, deadline_ms: int) -> None:
        """Sync helper called from the prefetch worker. Filters the
        already-in-RAM case (the whole point of prefetch is to PUT
        it in RAM), attempts L3 fetch, stores into RAM with TTL.
        Pulled out so unit tests can exercise it without spinning
        up the asyncio worker.

        Uses the public `HostStore.peek_in_ram` / `warm_from_ssd`
        APIs so we don't touch HostStore's private attrs and — more
        importantly — so the disk I/O step doesn't hold the store
        lock. The previous implementation reached into `_lock`,
        `_entries`, `_long_region`, `_spillover` directly, AND held
        `_lock` across the long-region's `get_bytes()` disk read;
        that meant every prefetch warm blocked every other get/set
        on the daemon for the duration of the seek. `warm_from_ssd`
        encapsulates the correct ordering (peek-under-lock,
        disk-read-without-lock, set-under-lock-via-public-API).

        Note: we use `peek_in_ram`, NOT `store.exists()` — the
        public `exists` reports True for keys that live on SSD
        only, which would short-circuit every prefetch into a no-op.
        """
        ttl_seconds = max(deadline_ms / 1000.0, 0.001)
        warmed = self._store.warm_from_ssd(
            model=model,
            compat_key=compat_key,
            key=key,
            ttl_seconds=ttl_seconds,
        )
        if warmed:
            self._prefetch_fetches_total += 1
            self._register_warmed_key(key)

    def _register_warmed_key(self, key: bytes) -> None:
        """Track recently-warmed keys so the next `get` on this
        daemon can be counted as a 'useful' prefetch. Bounded set
        with FIFO eviction at `_PREFETCH_WARMED_CAP`."""
        if key in self._prefetch_warmed_keys:
            # Move to most-recent — pop from order list, re-append.
            try:
                self._prefetch_warmed_order.remove(key)
            except ValueError:
                pass
            self._prefetch_warmed_order.append(key)
            return
        if len(self._prefetch_warmed_order) >= _PREFETCH_WARMED_CAP:
            oldest = self._prefetch_warmed_order.pop(0)
            self._prefetch_warmed_keys.discard(oldest)
        self._prefetch_warmed_keys.add(key)
        self._prefetch_warmed_order.append(key)

    def claim_warmed_key(self, key: bytes) -> bool:
        """Atomic test-and-clear: returns True if this key was warmed
        by the prefetch worker before now. Caller is expected to be
        the `get` dispatch — on True it increments the 'useful'
        prefetch counter. Pulling the key out of the warmed set on
        first read means a second get on the same key isn't double-
        counted."""
        if key in self._prefetch_warmed_keys:
            self._prefetch_warmed_keys.discard(key)
            try:
                self._prefetch_warmed_order.remove(key)
            except ValueError:
                pass
            self._prefetch_hits_useful_total += 1
            return True
        return False

    @property
    def prefetch_stats(self) -> dict[str, int]:
        """Operational snapshot of the prefetch worker. Surfaced via
        stats opcode or test introspection."""
        return {
            "hints_total": self._prefetch_hints_total,
            "fetches_total": self._prefetch_fetches_total,
            "dropped_total": self._prefetch_dropped_total,
            "hits_useful_total": self._prefetch_hits_useful_total,
            "warmed_inflight": len(self._prefetch_warmed_keys),
            "queue_depth": self._prefetch_queue.qsize() if self._prefetch_queue else 0,
        }

    def shutdown(self) -> None:
        """Trigger graceful shutdown from any task (no `await` here so
        signal handlers can call it)."""
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Connection handler
    # ------------------------------------------------------------------

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        peer = writer.get_extra_info("peername") or "<uds>"
        client_id = f"<unknown {peer}>"
        # Per-connection negotiation state. When True, GET/BatchGet
        # dispatch returns *Shared variants (offsets, no bytes); the
        # arena FD was sent via SCM_RIGHTS immediately after HelloAck.
        client_shared_arena = False
        # Per-connection id for save-side CopyFree lease ownership.
        # Assigned eagerly so SetReserve handlers can stamp it onto
        # the arena's `_reservations` map. Lock the counter so two
        # concurrent connections never collide.
        with self._connection_id_lock:
            connection_id = self._next_connection_id
            self._next_connection_id += 1
        try:
            # First frame must be Hello; let us learn the client_id for logs.
            first = await read_frame(reader)
            if isinstance(first, Hello):
                client_id = first.client_id
                logger.debug(
                    "client connected: %s (v%d, prefers_shared_arena=%s)",
                    client_id,
                    first.protocol_version,
                    first.prefers_shared_arena,
                )
                # Decide whether to enable shared arena for this client.
                # Conditions: (1) client opted in, (2) server has an
                # arena wired on the store. We trust the client's
                # IPC-namespace claim — checking would require
                # /proc/<pid>/ns/ipc inspection of the peer credentials,
                # which is gated on SO_PEERCRED + namespace ID
                # comparison. Phase 2 work. For now: if both sides
                # agreed, we send the FD; if it lands in a different
                # IPC namespace, the worker's mmap will fail visibly.
                arena = self._store.shared_arena
                send_shared = bool(first.prefers_shared_arena and arena is not None)
                ack = HelloAck(
                    server_id=self._server_id,
                    protocol_version=_PROTOCOL_VERSION,
                    shared_arena=arena.info.to_tuple() if send_shared else None,
                )
                await write_frame(writer, ack)
                if send_shared:
                    # Send the arena FD via SCM_RIGHTS on the same
                    # socket, RIGHT after the HelloAck frame. The
                    # client `read_frame`s the HelloAck (consuming all
                    # framed bytes), then immediately reads the
                    # ancillary message. The drain inside
                    # `send_fd_async` ensures HelloAck has been pushed.
                    try:
                        await send_fd_async(writer, arena.fd)
                        client_shared_arena = True
                    except Exception:
                        logger.exception(
                            "client %s: SCM_RIGHTS send failed — falling "
                            "back to inline-bytes responses",
                            client_id,
                        )
                        client_shared_arena = False
            else:
                # Allow no-hello mode for tiny clients (tests).
                await self._dispatch(
                    first,
                    writer,
                    client_shared_arena=False,
                    connection_id=connection_id,
                )

            while True:
                msg = await read_frame(reader)
                await self._dispatch(
                    msg,
                    writer,
                    client_shared_arena=client_shared_arena,
                    connection_id=connection_id,
                )
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            # Normal disconnect.
            pass
        except Exception:
            logger.exception("kvd connection from %s failed", client_id)
        finally:
            # Cancel-on-disconnect: drop every save lease this
            # connection still held so a worker crash doesn't leak
            # arena slots. No-op when the arena isn't wired or the
            # connection never used the lease path.
            arena = self._store.shared_arena
            if arena is not None:
                try:
                    cancelled = arena.cancel_connection_reservations(connection_id)
                    if cancelled:
                        logger.info(
                            "kvd connection %s (id=%d) dropped — cancelled "
                            "%d outstanding save lease(s)",
                            client_id,
                            connection_id,
                            cancelled,
                        )
                except Exception:
                    # Cleanup must not fail the close path.
                    logger.exception("kvd connection %s: cancel-on-disconnect failed", client_id)
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, BrokenPipeError):
                pass

    async def _dispatch(
        self,
        msg,
        writer: asyncio.StreamWriter,
        *,
        client_shared_arena: bool = False,
        connection_id: int = 0,
    ) -> None:
        """Route a single request message to the handler that produces
        the matching response. We never let an unhandled exception
        escape — return an ErrorMessage so the client can decide what
        to do next.

        ``client_shared_arena`` is set per-connection from the Hello
        handshake. When True AND the entry is arena-backed, GET/BatchGet
        return *Shared variants (offsets + version, no bytes); otherwise
        the existing inline-bytes responses. A mixed-mode store (some
        entries arena-backed, others inline) still works because the
        per-entry `slot_id` switches the per-response shape.
        """
        try:
            if isinstance(msg, Get):
                entry = self._store.get(msg.key, model=msg.model, compat_key=msg.compat_key)
                if entry is None:
                    await write_frame(writer, GetResponse(value=None, retention=None))
                else:
                    # Issue #20 item 3: count "useful" prefetch hits.
                    # The warmed-key set is touched only when a
                    # prefetch actually fired and the engine actually
                    # asks for the bytes — that's the bench-relevant
                    # signal in §9 Phase 3's exit criterion.
                    self.claim_warmed_key(msg.key)
                    if client_shared_arena and entry.slot_id >= 0:
                        # Shared-arena dispatch — return (offset, length,
                        # version). The client reads bytes from its
                        # own mmap; no payload in the response.
                        arena = self._store.shared_arena
                        meta = arena.get_slot_metadata(entry.slot_id)
                        if meta is None:
                            # Race: slot was evicted between get() and
                            # metadata read. Fall back to inline-bytes
                            # response with whatever we can resolve
                            # (likely empty; client treats as miss).
                            value = self._store.resolve_value(entry)
                            await write_frame(
                                writer,
                                GetResponse(value=value or None, retention=entry.retention),
                            )
                        else:
                            offset, length, version = meta
                            await write_frame(
                                writer,
                                GetSharedResponse(
                                    slot_offset=offset,
                                    length=length,
                                    version=version,
                                    retention=entry.retention,
                                    slot_size=arena.slot_size,
                                ),
                            )
                    else:
                        # Inline-bytes response. Materialize via
                        # resolve_value so arena-backed entries still
                        # work for clients that didn't opt in.
                        value = self._store.resolve_value(entry)
                        await write_frame(
                            writer,
                            GetResponse(value=value, retention=entry.retention),
                        )
            elif isinstance(msg, BatchGet):
                # Resolve every key in ONE pass via get_many, which reads the
                # long region with a single get_bytes_batch RPC instead of N
                # single get_bytes — collapsing N network round trips into one
                # on a distributed L4 backend. get_many bumps
                # gets_total / hits_total / misses_total per key, matching the
                # single-`Get` path.
                entries = self._store.get_many(msg.keys, model=msg.model, compat_key=msg.compat_key)
                if client_shared_arena and self._store.shared_arena is not None:
                    arena = self._store.shared_arena
                    offsets: list[int] = []
                    lengths: list[int] = []
                    versions: list[int] = []
                    retentions_shared: list[str | None] = []
                    for entry in entries:
                        if entry is None or entry.slot_id < 0:
                            # Miss OR inline-fallback entry. We can't
                            # carry inline bytes in a BatchGetSharedResponse,
                            # so for inline-fallback (rare — would only
                            # happen on arena rejection during set) we
                            # report miss; the client falls through to
                            # long-region path. This is a deliberate
                            # trade-off for the shared-arena fast path.
                            offsets.append(-1)
                            lengths.append(0)
                            versions.append(0)
                            retentions_shared.append(None)
                            continue
                        self.claim_warmed_key(entry.key)
                        meta = arena.get_slot_metadata(entry.slot_id)
                        if meta is None:
                            offsets.append(-1)
                            lengths.append(0)
                            versions.append(0)
                            retentions_shared.append(None)
                        else:
                            offset, length, version = meta
                            offsets.append(offset)
                            lengths.append(length)
                            versions.append(version)
                            retentions_shared.append(entry.retention)
                    await write_frame(
                        writer,
                        BatchGetSharedResponse(
                            offsets=offsets,
                            lengths=lengths,
                            versions=versions,
                            retentions=retentions_shared,
                            slot_size=arena.slot_size,
                        ),
                    )
                else:
                    values: list[bytes | None] = []
                    retentions: list[str | None] = []
                    for entry in entries:
                        if entry is None:
                            values.append(None)
                            retentions.append(None)
                        else:
                            # See `Get` handler: count "useful" prefetch
                            # hits on the batch path too (item #3).
                            self.claim_warmed_key(entry.key)
                            values.append(self._store.resolve_value(entry))
                            retentions.append(entry.retention)
                    await write_frame(
                        writer, BatchGetResponse(values=values, retentions=retentions)
                    )
            elif isinstance(msg, Set):
                accepted, reason = self._store.set(
                    msg.key,
                    msg.value,
                    retention=msg.retention,
                    model=msg.model,
                    compat_key=msg.compat_key,
                    metadata=msg.metadata,
                    ttl_seconds=msg.ttl_seconds,
                )
                await write_frame(writer, SetAck(accepted=accepted, reason=reason))
            elif isinstance(msg, BatchSet):
                # Loop over items calling `store.set` per key. The store
                # bumps `sets_total` per call, so the per-key counters
                # stay consistent with the single-`Set` path.
                n = len(msg.keys)
                # ttls_seconds is the only optional parallel array —
                # absent (None) means "no TTL on any item"; a list
                # must agree on length with the rest.
                ttls = msg.ttls_seconds
                ttl_arity_ok = ttls is None or len(ttls) == n
                if not (
                    len(msg.values) == n
                    and len(msg.retentions) == n
                    and len(msg.metadatas) == n
                    and ttl_arity_ok
                ):
                    await write_frame(
                        writer,
                        ErrorMessage(
                            code="batch_set_arity_mismatch",
                            message=(
                                f"BatchSet arrays disagree: "
                                f"keys={n} values={len(msg.values)} "
                                f"retentions={len(msg.retentions)} "
                                f"metadatas={len(msg.metadatas)} "
                                f"ttls_seconds={None if ttls is None else len(ttls)}"
                            ),
                        ),
                    )
                else:
                    accepted_list: list[bool] = []
                    reasons_list: list[str | None] = []
                    for i in range(n):
                        a, r = self._store.set(
                            msg.keys[i],
                            msg.values[i],
                            retention=msg.retentions[i],
                            model=msg.model,
                            compat_key=msg.compat_key,
                            metadata=msg.metadatas[i],
                            ttl_seconds=(None if ttls is None else ttls[i]),
                        )
                        accepted_list.append(a)
                        reasons_list.append(r)
                    await write_frame(
                        writer, BatchSetAck(accepted=accepted_list, reasons=reasons_list)
                    )
            elif isinstance(msg, SetReserve):
                # Save-side CopyFree, phase 1: reserve an arena slot
                # so the engine can H2D-copy bytes directly into the
                # shared mmap. No arena → reject with a stable
                # reason so the client falls back to inline Set.
                arena = self._store.shared_arena
                if arena is None:
                    await write_frame(
                        writer,
                        SetReserveResponse(
                            lease_token=0,
                            slot_id=-1,
                            payload_offset=0,
                            payload_max_size=0,
                            reason="no_arena",
                        ),
                    )
                else:
                    result = arena.reserve(msg.size, connection_id=connection_id)
                    if result is None:
                        # arena.reserve returns None on oversize OR
                        # arena-full-no-evictable. Pick the right
                        # reason by inspecting the current slot grid.
                        if (
                            arena.slot_size > 0
                            and msg.size + 16 > arena.slot_size  # _HEADER_BYTES = 16
                        ):
                            reason = "oversize"
                        elif msg.size > arena.capacity_bytes:
                            reason = "oversize"
                        else:
                            reason = "arena_full_no_evictable"
                        await write_frame(
                            writer,
                            SetReserveResponse(
                                lease_token=0,
                                slot_id=-1,
                                payload_offset=0,
                                payload_max_size=0,
                                reason=reason,
                            ),
                        )
                    else:
                        lease_token, slot_id, payload_offset = result
                        payload_max_size = arena.slot_size - 16
                        await write_frame(
                            writer,
                            SetReserveResponse(
                                lease_token=lease_token,
                                slot_id=slot_id,
                                payload_offset=payload_offset,
                                payload_max_size=payload_max_size,
                                reason="",
                            ),
                        )
            elif isinstance(msg, SetCommit):
                # Phase 2: finalize the lease. We expect the engine
                # has written its bytes into the slot at the offset
                # we handed back at reserve time.
                arena = self._store.shared_arena
                if arena is None:
                    await write_frame(
                        writer,
                        SetCommitResponse(accepted=False, reason="no_arena"),
                    )
                else:
                    # Sanity-check retention. Convert "default" to
                    # short — caller doesn't know our policy.
                    retention_str = msg.retention
                    if retention_str == "default" or retention_str == "":
                        retention_str = "short"
                    if retention_str not in _VALID_RETENTIONS:
                        await write_frame(
                            writer,
                            SetCommitResponse(accepted=False, reason="bad_retention"),
                        )
                    else:
                        ok, reason, overwritten = arena.commit_reservation(
                            msg.lease_token,
                            key=msg.key,
                            length=msg.length,
                            connection_id=connection_id,
                        )
                        if not ok:
                            await write_frame(
                                writer,
                                SetCommitResponse(accepted=False, reason=reason),
                            )
                        else:
                            composite = (msg.model, msg.compat_key, msg.key)
                            ttl_seconds = float(msg.ttl_seconds) if msg.ttl_seconds > 0 else None
                            try:
                                self._store.commit_arena_lease(
                                    composite,
                                    slot_id=arena.get_slot_for_key(msg.key) or -1,
                                    length=msg.length,
                                    retention=retention_str,
                                    ttl_seconds=ttl_seconds,
                                    overwritten_key=overwritten,
                                )
                                # `arena.get_slot_for_key` above
                                # touched LRU — that's intentional
                                # since the commit is also an MRU
                                # event for the slot.
                                await write_frame(
                                    writer,
                                    SetCommitResponse(accepted=True, reason=""),
                                )
                            except Exception as exc:
                                logger.exception(
                                    "commit_arena_lease failed lease=%d key=%s",
                                    msg.lease_token,
                                    msg.key.hex()[:16],
                                )
                                await write_frame(
                                    writer,
                                    SetCommitResponse(
                                        accepted=False,
                                        reason=f"commit_failed:{str(exc)[:60]}",
                                    ),
                                )
            elif isinstance(msg, SetCancel):
                arena = self._store.shared_arena
                if arena is not None:
                    arena.cancel_reservation(msg.lease_token, connection_id=connection_id)
                # Always ack — cancel is idempotent at the wire level
                # so the engine's retry-safety net stays simple.
                await write_frame(writer, SetCancelResponse())
            elif isinstance(msg, PrefetchHint):
                # Issue #20 item 3 — speculative L3 prefetch.
                # Fire-and-forget: no response, the router doesn't
                # wait. We bump the hint counter, then enqueue each
                # key onto the bounded prefetch queue. Drops on
                # overflow are counted (operator visibility) but
                # never block the read loop.
                self._prefetch_hints_total += 1
                if self._prefetch_queue is not None:
                    for k in msg.keys:
                        try:
                            self._prefetch_queue.put_nowait(
                                (k, msg.model, msg.compat_key, msg.deadline_ms)
                            )
                        except asyncio.QueueFull:
                            self._prefetch_dropped_total += 1
                # NO `await write_frame(...)` — wire contract is
                # request-only.
            elif isinstance(msg, Exists):
                present = self._store.exists(msg.keys, model=msg.model, compat_key=msg.compat_key)
                await write_frame(writer, ExistsResponse(present=present))
            elif isinstance(msg, Clear):
                count = self._store.clear(model=msg.model, compat_key=msg.compat_key)
                await write_frame(writer, ClearAck(cleared_entries=count))
            elif isinstance(msg, Stats):
                s = self._store.stats
                await write_frame(
                    writer,
                    StatsResponse(
                        entries=s.entries,
                        host_bytes=s.host_bytes,
                        spillover_bytes=self._store.spillover_bytes,
                        long_bytes=self._store.long_bytes,
                        gets_total=s.gets_total,
                        sets_total=s.sets_total,
                        hits_total=s.hits_total,
                        misses_total=s.misses_total,
                        evictions_total=s.evictions_total,
                    ),
                )
            elif isinstance(msg, Hello):
                # Re-hello mid-stream: just ack again. We DON'T re-
                # negotiate shared-arena here — the FD was sent at
                # the original handshake; sending another would
                # double-mmap on the client side. The client should
                # not re-Hello once it has an arena.
                await write_frame(
                    writer,
                    HelloAck(server_id=self._server_id, protocol_version=_PROTOCOL_VERSION),
                )
            else:
                await write_frame(
                    writer,
                    ErrorMessage(
                        code="bad_op", message=f"unsupported message type: {type(msg).__name__}"
                    ),
                )
        except Exception as exc:
            logger.exception("dispatch failed for %s", type(msg).__name__)
            try:
                await write_frame(
                    writer, ErrorMessage(code="internal_error", message=str(exc)[:200])
                )
            except (OSError, BrokenPipeError):
                pass


# ----------------------------------------------------------------------
# CLI entrypoint — `python -m infera.kvd`
# ----------------------------------------------------------------------


def _parse_size(value: str) -> int:
    """`32G` → 32 * 1024**3, `512M` → 512 * 1024**2. Also accepts plain int."""
    value = value.strip()
    if not value:
        raise ValueError("empty size")
    suffix = value[-1].upper()
    if suffix in {"K", "M", "G", "T"}:
        n = float(value[:-1])
        mult = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[suffix]
        return int(n * mult)
    return int(value)


def _parse_size_or_auto(value: str) -> int | str:
    """Like `_parse_size`, but the literal string ``auto`` is passed
    through unchanged so the main() can resolve it to another arg's
    value at startup. Used for `--shared-arena-bytes` where the
    sensible default is "match `--max-bytes`"."""
    if isinstance(value, str) and value.strip().lower() == "auto":
        return "auto"
    return _parse_size(value)


def _load_yaml_config(path: str) -> dict:
    """Minimal YAML loader for L4 backend configs. Expands ``${ENV}``
    references in string values. Uses PyYAML if available, else a tiny
    flat key:value fallback (sufficient for the documented config shape).
    """
    import os as _os
    import re as _re

    def _expand(v):
        if isinstance(v, str):
            return _re.sub(r"\$\{([^}]+)\}", lambda m: _os.environ.get(m.group(1), ""), v)
        return v

    try:
        import yaml  # type: ignore

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
    except ImportError:
        # Flat fallback: "key: value" lines, no nesting except a single
        # 'connector_extra:' block of indented 'k: v' pairs.
        raw = {}
        cur_block = None
        with open(path) as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if line.startswith((" ", "\t")) and cur_block is not None:
                    k, _, v = line.strip().partition(":")
                    raw[cur_block][k.strip()] = v.strip().strip('"').strip("'")
                    continue
                k, _, v = line.partition(":")
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if v == "":
                    raw[k] = {}
                    cur_block = k
                else:
                    raw[k] = v
                    cur_block = None

    def _walk(obj):
        if isinstance(obj, dict):
            return {k: _walk(_expand(v)) for k, v in obj.items()}
        return _expand(obj)

    return _walk(raw)


def _l4_env_or(raw: dict, yaml_key: str, env_suffix: str, default, cast=str):
    """Resolve an L4 config value: ``INFERA_L4_<ENV_SUFFIX>`` env wins, then
    the YAML key, then the default. Env-first keeps k8s deployments fully
    declarative (ConfigMap / Secret / downward-API) — no mounted file to
    edit, and per-pod values (POD_IP) can't live in a shared ConfigMap."""
    val = os.environ.get(f"INFERA_L4_{env_suffix}")
    if val is not None and val != "":
        return cast(val)
    rv = raw.get(yaml_key)
    if rv is not None and rv != "":
        return cast(rv)
    return default


def _build_distributed_long_region(backend: str, args):
    """Construct a Mooncake or LMCache long region. Config comes from the
    optional ``--{backend}-config`` YAML and/or ``INFERA_L4_*`` env vars
    (env wins — see ``_l4_env_or``). Raises SystemExit with an actionable
    message on misconfiguration."""
    if backend == "mooncake":
        cfg_path = getattr(args, "mooncake_config", None)
        from infera.kvd.mooncake_long_region import (
            MooncakeStoreConfig,
            MooncakeStoreLongRegion,
        )

        raw = _load_yaml_config(cfg_path) if cfg_path else {}
        master_address = _l4_env_or(raw, "master_address", "MASTER_ADDRESS", "")
        cluster_id = _l4_env_or(raw, "cluster_id", "CLUSTER_ID", "")
        missing = [
            n for n, v in (("master_address", master_address), ("cluster_id", cluster_id)) if not v
        ]
        if missing:
            raise SystemExit(
                f"kvd: mooncake config missing required keys {missing} — set them "
                f"in --mooncake-config or via INFERA_L4_MASTER_ADDRESS / "
                f"INFERA_L4_CLUSTER_ID"
            )
        # local_hostname must be a routable host:port. In k8s the only source
        # of the pod's own routable address is the downward API (POD_IP) — a
        # shared ConfigMap can't carry it, and gethostname() returns the
        # non-routable pod name. Derive POD_IP:port when not set explicitly.
        local_hostname = _l4_env_or(raw, "local_hostname", "LOCAL_HOSTNAME", "")
        if not local_hostname:
            pod_ip = os.environ.get("POD_IP", "").strip()
            if pod_ip:
                port = os.environ.get("INFERA_L4_LOCAL_PORT", "14001").strip() or "14001"
                local_hostname = f"{pod_ip}:{port}"
        config = MooncakeStoreConfig(
            master_address=master_address,
            cluster_id=cluster_id,
            metadata_server=_l4_env_or(raw, "metadata_server", "METADATA_SERVER", ""),
            protocol=_l4_env_or(raw, "protocol", "PROTOCOL", "tcp"),
            device_name=_l4_env_or(raw, "device_name", "DEVICE_NAME", ""),
            local_hostname=local_hostname,
            global_segment_size=_l4_env_or(
                raw, "global_segment_size", "GLOBAL_SEGMENT_SIZE", 1 << 30, int
            ),
            local_buffer_size=_l4_env_or(
                raw, "local_buffer_size", "LOCAL_BUFFER_SIZE", 1 << 28, int
            ),
            max_value_bytes=_l4_env_or(
                raw, "max_value_bytes", "MAX_VALUE_BYTES", 64 * 1024 * 1024, int
            ),
        )
        return MooncakeStoreLongRegion(config)

    if backend == "lmcache":
        cfg_path = getattr(args, "lmcache_config", None)
        from infera.kvd.lmcache_long_region import (
            LMCacheRemoteConfig,
            LMCacheRemoteLongRegion,
        )

        raw = _load_yaml_config(cfg_path) if cfg_path else {}
        remote_url = _l4_env_or(raw, "remote_url", "REMOTE_URL", "")
        if not remote_url:
            raise SystemExit(
                "kvd: lmcache config missing required key remote_url — set it in "
                "--lmcache-config or via INFERA_L4_REMOTE_URL"
            )
        config = LMCacheRemoteConfig(
            remote_url=remote_url,
            prefix=_l4_env_or(raw, "prefix", "PREFIX", "infera"),
            serde=_l4_env_or(raw, "serde", "SERDE", "naive"),
            connector_extra=dict(raw.get("connector_extra", {}) or {}),
            max_value_bytes=_l4_env_or(
                raw, "max_value_bytes", "MAX_VALUE_BYTES", 64 * 1024 * 1024, int
            ),
            rpc_timeout_s=_l4_env_or(raw, "rpc_timeout_s", "RPC_TIMEOUT_S", 30.0, float),
        )
        return LMCacheRemoteLongRegion(config)

    raise SystemExit(f"kvd: unknown --long-backend {backend!r}")


def _resolve_io_mode(args, probe_path: str | Path) -> tuple[bool, str]:
    """Resolve the effective ``o_direct`` decision for the L3 region.

    Priority order:

      1. ``INFERA_KVD_IO_MODE`` env var (``direct`` / ``buffered``)
         — operator escape hatch for production overrides without
         touching deployment manifests.
      2. ``--io-mode {direct,buffered}`` CLI flag — explicit operator
         intent.
      3. ``--io-mode auto`` (default) — call
         ``storage_classify.pick_io_mode(probe_path)`` and log the
         decision + rationale. Includes a full ``format_decision``
         block so the chosen io_mode is auditable from the kvd log.

    Returns ``(o_direct, source_tag)`` where ``source_tag`` is a short
    human label ("auto", "explicit", "env") used in the INFO log.
    """
    import os

    from infera.kvd.storage_classify import format_decision, pick_io_mode

    env = os.environ.get("INFERA_KVD_IO_MODE", "").strip().lower()
    if env in ("direct", "buffered"):
        o_direct = env == "direct"
        logger.info(
            "[kvd] L3 io_mode: %s (INFERA_KVD_IO_MODE=%s)",
            "DIRECT" if o_direct else "BUFFERED",
            env,
        )
        return o_direct, "env"

    mode = getattr(args, "io_mode", "auto")
    if mode == "direct":
        logger.info("[kvd] L3 io_mode: DIRECT (explicit --io-mode direct)")
        return True, "explicit"
    if mode == "buffered":
        logger.info("[kvd] L3 io_mode: BUFFERED (explicit --io-mode buffered)")
        return False, "explicit"

    # auto — probe and log a full multi-line decision block.
    probe = Path(probe_path)
    try:
        o_direct, _rationale = pick_io_mode(probe)
        for line in format_decision(probe).splitlines():
            logger.info("[kvd] %s", line)
    except Exception as exc:  # pragma: no cover — defensive
        logger.warning(
            "[kvd] L3 io_mode auto-classify on %s raised %s; falling back to BUFFERED",
            probe,
            exc,
        )
        return False, "auto"
    return o_direct, "auto"


async def _main_async(args) -> None:
    # Optional SSD tiers — wired only when the operator provides paths.
    # Either or both can be enabled independently.
    from infera.kvd.shared_arena import SharedArena
    from infera.kvd.ssd import LongStorageRegion
    from infera.kvd.store import HostStore

    spillover = None
    long_region = None
    shared_arena = None
    # Resolve `auto` → match `--max-bytes`. The arena IS the RAM tier
    # in that mode; operators who want to opt out pass
    # `--shared-arena-bytes 0` explicitly.
    arena_bytes: int
    if args.shared_arena_bytes == "auto":
        arena_bytes = args.max_bytes
        logger.info(
            "shared arena: --shared-arena-bytes=auto → using --max-bytes (%.2f GiB)",
            arena_bytes / (1024**3),
        )
    else:
        arena_bytes = int(args.shared_arena_bytes)
    if arena_bytes > 0:
        shared_arena = SharedArena(
            arena_bytes,
            name="infera-kvd-arena",
            pin_memory=args.shared_arena_pin,
            hugetlb=args.shared_arena_hugetlb,
        )
        logger.info(
            "shared arena enabled: %.2f GiB, pin=%s, hugetlb=%s",
            arena_bytes / (1024**3),
            args.shared_arena_pin,
            shared_arena.hugetlb_active,
        )
    # Spillover tier removed from the config surface — single disk tier only.
    spillover = None

    # ------------------------------------------------------------------
    # L4 distributed long-backend selection (task #255).
    # --long-backend {tablespace | mooncake | lmcache}. Default
    # 'tablespace' keeps the existing local NVMe/NFS behavior below.
    # The distributed backends are mutually exclusive with --long-path(s):
    # they ARE the long region, sourced from a remote cluster.
    # ------------------------------------------------------------------
    long_backend = getattr(args, "long_backend", "tablespace")
    if long_backend in ("mooncake", "lmcache"):
        if args.long_path or args.long_paths:
            raise SystemExit(
                f"kvd: --long-backend {long_backend} is mutually exclusive with "
                "--long-path / --long-paths (the distributed backend is the "
                "long region; there's no local striping to configure)."
            )
        long_region = _build_distributed_long_region(long_backend, args)
        long_region.start()
        store = HostStore(
            max_bytes=args.max_bytes,
            spillover=spillover,
            long_region=long_region,
            shared_arena=shared_arena,
        )
        logger.info(
            "[kvd] L4 long-backend: %s (distributed). Local striping skipped.",
            long_backend,
        )
        server = KvdServer(socket_path=args.socket, max_bytes=args.max_bytes, store=store)
        await server.start()
        await server.serve_forever()
        return

    if args.long_path and args.long_paths:
        raise SystemExit(
            "kvd: --long-path and --long-paths are mutually exclusive; "
            "use --long-path for single-device (existing behavior) or "
            "--long-paths for multi-device striping (Phase 3)."
        )
    if args.long_paths and not args.use_tablespace:
        raise SystemExit(
            "kvd: --long-paths requires --use-tablespace (the striped "
            "region shards over TablespaceLongRegion instances; the "
            "legacy LongStorageRegion is not supported)."
        )
    if args.long_paths:
        # Striped long region — N shards, one per mount point, hash-routed.
        from infera.kvd.striped_long_region import StripedLongRegion
        from infera.kvd.tablespace import TablespaceLongRegion

        paths = [p.strip() for p in args.long_paths.split(",") if p.strip()]
        if not paths:
            raise SystemExit("kvd: --long-paths must be a non-empty comma-separated list")
        # Resolution order: INFERA_KVD_IO_MODE > --io-mode (non-auto)
        # > legacy --tablespace-buffered-io > --tablespace-auto-detect-fs
        # > --io-mode auto (storage_classify probe).
        if args.tablespace_buffered_io:
            o_direct = False
            logger.info("[kvd] L3 io_mode: BUFFERED (explicit --tablespace-buffered-io)")
        elif args.tablespace_auto_detect_fs:
            # Defer to TablespaceLongRegion's fstype-based auto-detect
            # (legacy behavior — kept for backward compat with deploy
            # manifests that already use this flag).
            o_direct = None
        else:
            # Classify on the first shard's path. The expectation is
            # that all striping shards live on similar devices (typical:
            # 8× NVMe under /mnt/nvme{0..7}); the auto-probe is a guide.
            o_direct, _ = _resolve_io_mode(args, paths[0])
        if args.tablespace_flush_interval_ms is not None:
            flush_interval_ms = args.tablespace_flush_interval_ms
        elif args.tablespace_auto_detect_fs:
            flush_interval_ms = None
        else:
            flush_interval_ms = 0
        per_shard_bytes = max(args.long_bytes // len(paths), args.tablespace_container_bytes)
        shards = [
            TablespaceLongRegion(
                p,
                per_shard_bytes,
                slot_bytes=args.tablespace_slot_bytes,
                container_bytes=args.tablespace_container_bytes,
                o_direct=o_direct,
                flush_interval_ms=flush_interval_ms,
            )
            for p in paths
        ]
        long_region = StripedLongRegion(
            shards,
            workers_per_shard=args.long_workers_per_shard,
        )
        long_region.start()
        logger.info(
            "striped long region: %d shards × %d bytes/shard = %d total, "
            "%d workers per shard (%d-way pread fanout)",
            len(paths),
            per_shard_bytes,
            per_shard_bytes * len(paths),
            args.long_workers_per_shard,
            len(paths) * args.long_workers_per_shard,
        )
    elif args.long_path:
        if args.use_tablespace:
            # Phase B: pre-allocated container files + bitset allocator +
            # append-only journal. Bounded file count, scales better
            # past ~100K entries than the file-per-block layout.

            # Resolution order for o_direct + flush_interval_ms:
            #   1. INFERA_KVD_IO_MODE env var — operator escape hatch.
            #   2. Explicit --tablespace-buffered-io flag — legacy
            #      backward-compat; treated as "I really mean buffered".
            #   3. --tablespace-auto-detect-fs — pass None down, let the
            #      region probe fstype at start() and pick per-backend
            #      defaults.
            #   4. --io-mode (default `auto`) — call storage_classify
            #      to pick based on the underlying device transport.
            #      Explicit `--io-mode direct/buffered` short-circuits.
            #
            # The `--io-mode auto` default is intentional (May 2026
            # rebrand): O_DIRECT was the unconditional choice and lost
            # to buffered on SATA + NFS-low-nconnect.
            if args.tablespace_buffered_io:
                o_direct = False
                logger.info("[kvd] L3 io_mode: BUFFERED (explicit --tablespace-buffered-io)")
            elif args.tablespace_auto_detect_fs:
                o_direct = None
            else:
                o_direct, _ = _resolve_io_mode(args, args.long_path)

            if args.tablespace_flush_interval_ms is not None:
                flush_interval_ms = args.tablespace_flush_interval_ms
            elif args.tablespace_auto_detect_fs:
                flush_interval_ms = None
            else:
                flush_interval_ms = 0

            if args.tablespace_pools:
                # Multi-pool mode (Phase B.2+). One
                # kvd serves heterogeneous block sizes (SGLang 64K +
                # vLLM packed 1M) without per-pool slot waste.
                from infera.kvd.tablespace_multipool import (
                    MultiPoolTablespaceLongRegion,
                    parse_pools_spec,
                )

                pools = parse_pools_spec(
                    args.tablespace_pools,
                    default_max_bytes_per_pool=args.long_bytes
                    // max(args.tablespace_pools.count(",") + 1, 1),
                )
                long_region = MultiPoolTablespaceLongRegion(
                    args.long_path,
                    pools=pools,
                    container_bytes=args.tablespace_container_bytes,
                    o_direct=o_direct,
                    flush_interval_ms=flush_interval_ms,
                )
            else:
                # Single-pool default (backward-compatible).
                from infera.kvd.tablespace import TablespaceLongRegion

                long_region = TablespaceLongRegion(
                    args.long_path,
                    args.long_bytes,
                    slot_bytes=args.tablespace_slot_bytes,
                    container_bytes=args.tablespace_container_bytes,
                    o_direct=o_direct,
                    flush_interval_ms=flush_interval_ms,
                )
        else:
            long_region = LongStorageRegion(args.long_path, args.long_bytes)
        long_region.start()

    # Startup L3 storage self-check — log write/read GB/s on the configured
    # long-path under kvd's resolved io_mode/workers/chunk, so a slow mount is
    # visible at boot instead of via degraded TTFT later. On by default;
    # disable with INFERA_KVD_STORAGE_SELFCHECK=0. Best-effort, never fatal.
    _sc_path = args.long_path or (
        args.long_paths.split(",")[0].strip() if args.long_paths else None
    )
    if _sc_path:
        try:
            from infera.kvd.storage_selfcheck import run_storage_selfcheck

            run_storage_selfcheck(_sc_path)
        except Exception as _sc_exc:  # pragma: no cover — never block startup
            logger.warning("[kvd] storage self-check wiring error (non-fatal): %s", _sc_exc)

    store = HostStore(
        max_bytes=args.max_bytes,
        spillover=spillover,
        long_region=long_region,
        shared_arena=shared_arena,
    )

    server = KvdServer(socket_path=args.socket, max_bytes=args.max_bytes, store=store)
    await server.start()
    await server.serve_forever()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="infera-kvd node-local KV cache daemon")
    parser.add_argument(
        "--socket",
        default="/var/run/infera-kvd.sock",
        help="Unix domain socket path (default: /var/run/infera-kvd.sock)",
    )
    parser.add_argument(
        "--max-bytes",
        type=_parse_size,
        default=_parse_size("8G"),
        help="Host RAM budget in bytes. Accepts suffixes K/M/G/T. Default: 8G.",
    )
    # NOTE: the spillover SSD tier has been removed from the config surface
    # (single disk tier = long region only; see feat/kvd-single-tier-simplify).
    # --spillover-path / --spillover-bytes are no longer accepted; the
    # SpilloverRegion code remains internally for now and will be deleted in a
    # follow-up. The disk tier is configured via --long-path / --long-bytes.
    parser.add_argument(
        "--long-path",
        default=None,
        help="Directory for the long-storage SSD region (Phase 4.0). "
        "When set, long-retention SETs are write_through to this region "
        "with fsync, and the daemon recovers the index from a persistent "
        "manifest on startup.",
    )
    parser.add_argument(
        "--long-paths",
        default=None,
        help="Comma-separated list of mount points for a striped "
        "long-region. Each path hosts its own "
        "complete TablespaceLongRegion; keys are hash-routed (blake2b) "
        "to one shard. Reads fanout across shards via a ThreadPoolExecutor "
        "so N inodes on N NVMe devices give real parallelism (single "
        "io_uring on one device only got 0.94×). Example: "
        "`/mnt/nvme0/kvd,/mnt/nvme1/kvd,...,/mnt/nvme7/kvd`. Mutually "
        "exclusive with --long-path. Requires --use-tablespace. "
        "--long-bytes is the TOTAL budget; per-shard budget is "
        "long_bytes // N.",
    )
    parser.add_argument(
        "--long-bytes",
        type=_parse_size,
        default=_parse_size("32G"),
        help="Long-storage region size budget. Only used when --long-path "
        "or --long-paths is set. With --long-paths, divided evenly across "
        "shards (long_bytes // N per shard). Default: 32G.",
    )
    parser.add_argument(
        "--long-backend",
        choices=("tablespace", "mooncake", "lmcache"),
        default="tablespace",
        help="Long-region backend (task #255). 'tablespace' "
        "(default) is the local NVMe/NFS striped region. 'mooncake' and "
        "'lmcache' are distributed L4 backends — mutually exclusive with "
        "--long-path/--long-paths. Mooncake needs --mooncake-config; "
        "lmcache needs --lmcache-config.",
    )
    parser.add_argument(
        "--mooncake-config",
        default=None,
        help="Path to a Mooncake Store YAML config (master_address, "
        "cluster_id, metadata_server, protocol, device_name, "
        "local_hostname, global_segment_size, local_buffer_size). "
        "Required when --long-backend mooncake.",
    )
    parser.add_argument(
        "--lmcache-config",
        default=None,
        help="Path to an LMCache remote-backend YAML config (remote_url, "
        "prefix, serde, connector_extra). Required when "
        "--long-backend lmcache.",
    )
    parser.add_argument(
        "--long-workers-per-shard",
        type=int,
        default=8,
        help="Intra-shard parallelism for the striped long region. Each "
        "shard's batched read fans out across this many sub-workers so "
        "concurrent reads inside one shard pipeline at the kernel level "
        "(get_bytes releases its lock before pread). With N shards and "
        "W workers/shard, get_bytes_batch issues an N×W-way parallel "
        "pread storm. Only used when --long-paths is set. Default: 8 "
        "(empirical knee of the throughput curve on 8-NVMe; pass 1 for "
        "the pre-sub-pool behavior).",
    )
    # Phase B: tablespace-pattern long region (pre-allocated container
    # files + bitset allocator + journal). Opt-in for now; the default
    # remains the file-per-block LongStorageRegion (Phase 4.0).
    parser.add_argument(
        "--use-tablespace",
        action="store_true",
        help="Use the tablespace-pattern long region instead of "
        "file-per-block. Bounded file count + better scaling past "
        "~100K entries. "
        "Only takes effect when --long-path is set.",
    )
    parser.add_argument(
        "--tablespace-slot-bytes",
        type=_parse_size,
        default=_parse_size("64K"),
        help="Fixed slot size for the tablespace long region. Must be "
        "≥ typical block size. Values larger than this are rejected. "
        "Default: 64K (matches TP=2 MiniMax-M2.5 page size).",
    )
    parser.add_argument(
        "--tablespace-container-bytes",
        type=_parse_size,
        default=_parse_size("1G"),
        help="Container file size for the tablespace long region. "
        "max_bytes / container_bytes = file count (kept small to avoid "
        "the FS metadata thrash that motivated this region). Default: 1G.",
    )
    parser.add_argument(
        "--tablespace-flush-interval-ms",
        type=int,
        default=None,
        help="Group-commit window. When > 0, tablespace PUTs skip "
        "inline fsync; a background thread fsyncs every interval. "
        "Big win on high-fsync-cost backends (NFS ~1.2 ms/fsync → "
        "10× write throughput at 10 ms window) at the cost of a "
        "bounded durability window (≤interval of writes lost on "
        "crash). Default (no flag): 0 = inline fsync (safe on local "
        "NVMe). When --tablespace-auto-detect-fs is on, the default "
        "comes from the per-fstype table instead. Recommended values: "
        "NVMe → 0, NFS / NAS → 10-50, 3FS RDMA → 5-10.",
    )
    parser.add_argument(
        "--tablespace-pools",
        default=None,
        help="Multi-pool tablespace spec — comma-separated slot sizes "
        "with optional weights. Examples: "
        "`64K,1M` (two equal pools), `64K*1,1M*4,4M*1` (weighted 1:4:1). "
        "When set, --tablespace-slot-bytes is ignored and the long "
        "region routes values to the smallest pool that fits. Use this "
        "when one kvd serves heterogeneous engines (e.g. SGLang's 64 KB "
        "per-layer-page writes + vLLM's 1 MB packed-block writes). "
        "See tests/unit/kvd/test_tablespace_multipool.py.",
    )
    # O_DIRECT is the default for the tablespace long region: the
    # page cache double-bookkeeps a daemon that owns its own cache
    # policy via HostStore. Bench measured on MI355X showed the
    # buffered path costs ~262 MB of page cache per 256 MB written —
    # not acceptable for production long regions.
    parser.add_argument(
        "--tablespace-buffered-io",
        action="store_true",
        help="Disable O_DIRECT for the tablespace long region; fall "
        "back to buffered IO through the kernel page cache. Default "
        "is O_DIRECT on (saves ~2× RAM use, +27%% write throughput; "
        "loses the warm-page-cache 18× speedup that rarely fires on "
        "kvd's workload anyway). Use this flag when the underlying "
        "filesystem rejects O_DIRECT (tmpfs, some NFS configs) — "
        "startup will tell you to do so via a clear error message.",
    )
    # Storage-aware io_mode selection. See infera.kvd.storage_classify.
    # Supersedes the binary
    # --tablespace-buffered-io flag for fresh deployments; the legacy
    # flag still works and short-circuits to buffered when set.
    parser.add_argument(
        "--io-mode",
        choices=("auto", "direct", "buffered"),
        default="auto",
        help="Pick L3 io_mode (O_DIRECT vs buffered) for the long "
        "region. Default `auto` runs the storage-classifier probe at "
        "startup: findmnt + lsblk walks the mount through "
        "mdraid/LVM/dm-crypt down to the underlying device transport, "
        "then picks DIRECT for NVMe/SAS-SSD and BUFFERED for SATA-SSD, "
        "HDD, iSCSI, FC, NFS, tmpfs. Explicit `direct` or `buffered` "
        "skips the probe. Env var `INFERA_KVD_IO_MODE=direct|buffered` "
        "overrides this flag.",
    )
    parser.add_argument(
        "--tablespace-auto-detect-fs",
        action="store_true",
        help="Auto-detect the filesystem at --long-path (via findmnt) "
        "and pick per-backend defaults for --tablespace-buffered-io / "
        "--tablespace-flush-interval-ms. Decision matrix: ext4/xfs/btrfs "
        "→ O_DIRECT on, inline fsync; nfs/nfs4 → O_DIRECT on, 20 ms "
        "group commit; wekafs → buffered (writecache coalesces RDMA), "
        "20 ms group commit; unknown → safe fallback (buffered, "
        "inline). Explicit --tablespace-buffered-io or "
        "--tablespace-flush-interval-ms still wins. Selected values "
        "are logged at INFO.",
    )
    # Shared-memory arena: opt-in cross-process zero-copy KV transport.
    # When enabled, vLLM/SGLang clients that opt in at handshake
    # receive an FD to a memfd-backed pinned arena; subsequent GETs
    # return (offset, length, version) and the client reads bytes
    # directly from its own mmap. Per-block UDS cost drops from
    # ~12 ms (5 MB packed block) to ~50 us + the local mmap read.
    # Requires --ipc=host on docker (shared IPC namespace).
    parser.add_argument(
        "--shared-arena-bytes",
        type=_parse_size_or_auto,
        default="auto",
        help="Capacity of the shared-memory KV arena, in bytes "
        "(accepts K/M/G/T suffixes). When > 0, the host RAM tier is "
        "backed by a memfd-backed pinned arena that compatible "
        "vLLM/SGLang clients mmap directly — get() drops from ~12 ms "
        "to ~50 us per 5 MB block. Requires `--ipc=host` on docker. "
        "Default `auto` = same as `--max-bytes` (the RAM-tier IS "
        "the arena). Pass `0` to opt out and fall back to the "
        "legacy inline-bytes wire path.",
    )
    parser.add_argument(
        "--shared-arena-pin",
        action="store_true",
        default=True,
        help="Lock the shared arena's pages into RAM via mlock(2). "
        "Default ON. Falls back silently to unpinned if the user "
        "lacks RLIMIT_MEMLOCK headroom; the arena still works, just "
        "without page-locking (DMA stages through unlocked memory).",
    )
    parser.add_argument(
        "--no-shared-arena-pin",
        action="store_false",
        dest="shared_arena_pin",
        help="Disable mlock on the shared arena (counterpart to "
        "--shared-arena-pin). Useful when RLIMIT_MEMLOCK is low.",
    )
    # 2 MB hugepage backing for the memfd. Default OFF — operators
    # must reserve pages via `sysctl -w vm.nr_hugepages=N` first; we
    # don't want kvd to fail to start on a system without hugepages
    # configured. Also overridable via the env
    # var `INFERA_KVD_ARENA_HUGETLB=1`.
    parser.add_argument(
        "--shared-arena-hugetlb",
        action="store_true",
        default=os.environ.get("INFERA_KVD_ARENA_HUGETLB", "").strip().lower()
        in ("1", "true", "yes", "on"),
        help="Back the shared arena's memfd with 2 MB hugepages "
        "(MFD_HUGETLB). Cuts TLB pressure on the host-side memcpy "
        "and smooths p99 first-touch faults. Default OFF — requires "
        "`vm.nr_hugepages` headroom; falls back to 4 KB with a WARN "
        "log if the allocation is refused. Env override: "
        "`INFERA_KVD_ARENA_HUGETLB=1`.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()
