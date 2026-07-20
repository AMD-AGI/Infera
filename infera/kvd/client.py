###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Async client for infera-kvd.

Engine adapters (SGLang's HiCacheStorage backend, vLLM's
KVConnectorBase_V1 impl) import this. The client wraps one persistent
Unix-socket connection and provides high-level async methods that
match the wire protocol one-to-one.

Connection lifecycle:
  - `connect()` opens the socket, sends Hello, awaits HelloAck.
  - `get/set/exists/clear/stats` send a request, await one response.
  - `close()` half-closes the stream and waits.

The client is **not** thread-safe — each engine instance should hold
its own client. If the engine has multiple worker threads issuing
cache ops concurrently, wrap the client in an async-aware queue or
spin up multiple connections.

Failure model:
  - `KvdConnectionError`: TCP/UDS-level errors (socket gone, can't
    bind, etc.). Caller's only choice is reconnect or give up.
  - `KvdProtocolError`: server sent ErrorMessage or unexpected
    response type. Recoverable for SET (downgrade retention,
    retry) but not for GET (means daemon-side bug).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pathlib import Path

from infera.kvd.shared_arena import (
    SharedArenaInfo,
    open_arena_view,
    read_slot_seqlock,
)
from infera.kvd.wire import (
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
    LookupTierResponse,
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


async def _sock_recv_exact(loop, sock, n: int) -> bytes:
    """Event-loop-friendly recv exactly `n` bytes from `sock`.
    Yields to the event loop on partial reads."""
    buf = bytearray()
    while len(buf) < n:
        chunk = await loop.sock_recv(sock, n - len(buf))
        if not chunk:
            raise ConnectionError(f"peer closed mid-frame ({len(buf)}/{n} bytes)")
        buf.extend(chunk)
    return bytes(buf)


async def _sock_recv_fd(loop, sock) -> int:
    """Event-loop-friendly recv_fd via SCM_RIGHTS. Uses sock_recv
    semantics: yields to the loop when the socket isn't readable.

    Internally calls `recvmsg`. The socket must be non-blocking
    (asyncio convention). We retry on BlockingIOError by waiting
    for the FD to be readable.
    """
    from infera.kvd.fd_passing import recv_fd

    while True:
        try:
            return recv_fd(sock)
        except BlockingIOError:
            # Wait for readable. Use sock_recv with 0 bytes to
            # block until the FD has data (asyncio's sock_recv
            # registers a reader callback; the dup trick from
            # fd_passing isn't needed here because we OWN this
            # socket, not asyncio).
            #
            # Note: `loop.sock_recv(sock, 0)` actually does NOT
            # block — it returns an empty bytes immediately. So
            # we use a 1-byte peek via MSG_PEEK, then loop back.
            # Or, simpler: use a future + add_reader.
            fut = loop.create_future()

            def _ready(fut=fut):  # bind loop var (B023)
                if not fut.done():
                    fut.set_result(None)

            loop.add_reader(sock, _ready)
            try:
                await fut
            finally:
                loop.remove_reader(sock)


class KvdConnectionError(IOError):
    """Underlying socket failed — can't reach the daemon."""


class KvdProtocolError(RuntimeError):
    """Daemon sent ErrorMessage or unexpected response type."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message


class KvdClient:
    """Async client. Use as `async with KvdClient(...) as client:` for
    automatic close, or manage `connect()`/`close()` yourself."""

    def __init__(
        self,
        socket_path: str | Path,
        *,
        client_id: str | None = None,
        prefer_shared_arena: bool = True,
        prefault_arena: bool | None = None,
    ) -> None:
        """``prefer_shared_arena``: when True (the default), the
        client opts into shared-arena negotiation. If the server is
        wired with a SharedArena AND supports the protocol bit, the
        handshake hands over the arena FD via SCM_RIGHTS; subsequent
        `get` / `batch_get` calls receive offsets+versions and read
        bytes directly from the mmap.

        Default True is silent for clients that don't care — they
        still call `.get()` and receive bytes (the client materializes
        them from the mmap). Clients that DO care about zero-copy
        should call `.get_view()` to receive a memoryview into the
        arena (no bytes copy).

        Set to False to force the inline-bytes path (backward-compat
        / debugging / tests). The wire then looks exactly like the
        pre-shared-arena protocol.

        ``prefault_arena``: when True (the default), pass
        ``MAP_POPULATE`` on the local mmap of the arena FD so all
        PTEs are walked eagerly. Eliminates first-touch page-fault
        spikes on the early get/set hot path.

        Cost vs benefit on the MI355X testbed: ~180 ms/GB at mmap time (≈2.9 s
        for a 16 GB production arena) buys a 9× drop in p50
        first-touch save latency (897 µs → 95 µs) and removes the
        p99 tail spike on the first ~100 saves. For long-running
        daemons the startup cost amortizes to zero; tail wins
        compound. ``MAP_POPULATE`` is a kernel hint — older kernels
        silently ignore it — so no operator config is required
        (contrast with ``MFD_HUGETLB``, which still defaults OFF
        because it needs ``vm.nr_hugepages`` pre-reservation).

        Default is None → consult ``INFERA_KVD_ARENA_PREFAULT``.
        Unset or any non-falsy value = ON. Set to ``0`` / ``false``
        / ``no`` / ``off`` to opt OUT."""
        self._socket_path = Path(socket_path)
        self._client_id = client_id or f"kvd-client-{uuid.uuid4().hex[:8]}"
        self._prefer_shared_arena = prefer_shared_arena
        self._prefault_arena = prefault_arena
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._server_id: str | None = None
        self._lock = asyncio.Lock()
        # Shared-arena state — populated on connect if negotiated.
        # `_arena_info` is the wire-tuple description.
        # `_arena_mmap` is our local mmap of the FD (read-only).
        # `_arena_mv` is a top-level memoryview (we slice it on each
        # GET so we never re-mmap).
        self._arena_info: SharedArenaInfo | None = None
        self._arena_fd: int | None = None
        self._arena_mmap = None
        self._arena_mv = None

    @property
    def server_id(self) -> str | None:
        """The kvd daemon's ID (returned in HelloAck). None before connect."""
        return self._server_id

    @property
    def shared_arena_negotiated(self) -> bool:
        """True iff `connect()` successfully negotiated a shared arena
        with the server AND mmap'd the FD. False otherwise (default
        inline-bytes path)."""
        return self._arena_mmap is not None

    async def __aenter__(self) -> KvdClient:
        await self.connect()
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the socket and exchange Hello frames.

        Shared-arena handshake (when both ends opt in):
          1. Client sends Hello(prefers_shared_arena=True).
          2. Server replies HelloAck(shared_arena=(size, slot, pid)).
          3. Server sends arena FD via SCM_RIGHTS on the same socket.
          4. Client mmaps the FD locally (read-only) and stores the
             view for subsequent gets.

        We do the handshake over a RAW socket (not asyncio streams)
        to avoid the asyncio-pump race: asyncio's transport reads
        bytes off the wire as soon as they arrive, which would
        consume the SCM_RIGHTS marker byte BEFORE we can call
        recvmsg with ancillary support. Raw socket gives us
        deterministic recv ordering. Once the handshake completes,
        we wrap the same FD in asyncio streams for subsequent IO.

        If any step fails, we fall back to the inline-bytes path
        silently — subsequent gets work as before, just slower.
        """
        import socket as socket_mod

        # Step 1: open a raw socket (we manage asyncio wrapping ourselves
        # after the handshake completes). Non-blocking so we can use
        # loop.sock_* without blocking the event loop.
        sock = socket_mod.socket(socket_mod.AF_UNIX, socket_mod.SOCK_STREAM)
        sock.setblocking(False)
        loop = asyncio.get_running_loop()
        try:
            await loop.sock_connect(sock, str(self._socket_path))
        except OSError as exc:
            sock.close()
            raise KvdConnectionError(
                f"could not connect to kvd at {self._socket_path}: {exc}"
            ) from exc

        # Step 2: send Hello + receive HelloAck over the raw socket
        # using `sock_sendall` / `sock_recv` (event-loop friendly).
        try:
            from infera.kvd.wire import LENGTH_BYTEORDER, LENGTH_PREFIX_BYTES, decode, encode

            hello_frame = encode(
                Hello(
                    client_id=self._client_id,
                    prefers_shared_arena=self._prefer_shared_arena,
                )
            )
            await loop.sock_sendall(sock, hello_frame)

            # Read length-prefixed HelloAck.
            header = await _sock_recv_exact(loop, sock, LENGTH_PREFIX_BYTES)
            length = int.from_bytes(header, LENGTH_BYTEORDER)
            body = await _sock_recv_exact(loop, sock, length)
            resp = decode(body)
        except (ConnectionError, OSError, ValueError) as exc:
            sock.close()
            raise KvdConnectionError(f"kvd handshake failed: {exc}") from exc

        if not isinstance(resp, HelloAck):
            sock.close()
            raise KvdProtocolError("bad_handshake", f"expected HelloAck, got {type(resp).__name__}")
        self._server_id = resp.server_id

        # Step 3: if shared-arena was negotiated, recv the FD now via
        # an event-loop-friendly recv_fd. The ancillary message has
        # already been sent by the server (after HelloAck) so the
        # socket is readable.
        if resp.shared_arena is not None:
            try:
                info = SharedArenaInfo.from_tuple(resp.shared_arena)
                fd = await _sock_recv_fd(loop, sock)
                # Open the arena RW so the save-side CopyFree path
                # (lease + commit) can write directly into reserved
                # slots. Reads are still seqlock-protected; the server
                # stamps the slot header so writers honor the contract.
                # ``prefault`` is None unless the operator overrode it
                # via constructor arg. Default behaviour is ON; the
                # env var ``INFERA_KVD_ARENA_PREFAULT=0`` opts out
                # (see ``open_arena_view`` docstring for rationale).
                mm, mv = open_arena_view(
                    fd,
                    info.arena_size,
                    writable=True,
                    prefault=self._prefault_arena,
                )
                self._arena_info = info
                self._arena_fd = fd
                self._arena_mmap = mm
                self._arena_mv = mv
                logger.info(
                    "kvd client %s: shared arena negotiated (%.2f GiB, slot=%d, server_pid=%d)",
                    self._client_id,
                    info.arena_size / (1024**3),
                    info.slot_size,
                    info.server_pid,
                )
            except Exception as exc:
                logger.warning(
                    "kvd client %s: shared-arena setup failed (%s) — "
                    "falling back to inline-bytes responses",
                    self._client_id,
                    exc,
                )
                self._arena_info = None
                self._arena_fd = None
                self._arena_mmap = None
                self._arena_mv = None

        # Step 4: wrap the raw socket in asyncio streams for subsequent
        # framed IO. The handshake is done; asyncio can take over.
        try:
            reader, writer = await asyncio.open_unix_connection(sock=sock)
        except OSError as exc:
            sock.close()
            raise KvdConnectionError(
                f"asyncio wrap of post-handshake socket failed: {exc}"
            ) from exc
        self._reader = reader
        self._writer = writer
        logger.debug(
            "kvd client %s connected to server %s (shared_arena=%s)",
            self._client_id,
            self._server_id,
            self.shared_arena_negotiated,
        )

    async def close(self) -> None:
        if self._writer is None and self._arena_mmap is None:
            return
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except (OSError, BrokenPipeError):
                pass
        # Release the shared arena resources LAST so any in-flight
        # `get_view` callers have already finished. The memoryview
        # release ordering matters: release `_arena_mv` (frees the
        # exported pointer count) BEFORE `mm.close()`, then close
        # the FD.
        if self._arena_mv is not None:
            try:
                self._arena_mv.release()
            except (BufferError, ValueError):
                pass
            self._arena_mv = None
        if self._arena_mmap is not None:
            try:
                self._arena_mmap.close()
            except (BufferError, ValueError):
                pass
            self._arena_mmap = None
        if self._arena_fd is not None:
            import os

            try:
                os.close(self._arena_fd)
            except OSError:
                pass
            self._arena_fd = None
        self._arena_info = None
        self._reader = None
        self._writer = None
        self._server_id = None

    # ------------------------------------------------------------------
    # KV operations
    # ------------------------------------------------------------------

    async def get(self, key: bytes, *, model: str = "", compat_key: str = "") -> bytes | None:
        """Return the block bytes, or None on miss.

        When the client negotiated a shared arena AND the server's
        entry is arena-backed, the response carries (offset, length,
        version) and we read bytes from our local mmap (zero-copy
        memoryview, then materialized to bytes for backward compat).
        Otherwise the response carries inline bytes.

        Callers that want zero-copy explicitly should use
        `get_view()` — same contract but returns a memoryview into
        the arena (no bytes copy)."""
        resp = await self._round_trip(Get(key=key, model=model, compat_key=compat_key))
        if isinstance(resp, GetSharedResponse):
            mv = self._materialize_shared(resp)
            return bytes(mv) if mv is not None else None
        if not isinstance(resp, GetResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return resp.value

    async def get_view(
        self, key: bytes, *, model: str = "", compat_key: str = ""
    ) -> memoryview | bytes | None:
        """Same as `get` but returns a memoryview into the arena
        when shared-arena is active (zero-copy). Falls back to bytes
        when the server returns an inline-bytes response (no arena).

        Callers must consume the memoryview before the arena slot
        can be overwritten — copy into a tensor or `bytes(mv)`
        immediately if you need to hold onto it. The client's torn-
        read detection means the slice is consistent at the moment
        of return, but a SECOND read of the same slice after another
        SET to the same key is undefined."""
        resp = await self._round_trip(Get(key=key, model=model, compat_key=compat_key))
        if isinstance(resp, GetSharedResponse):
            return self._materialize_shared(resp)
        if not isinstance(resp, GetResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return resp.value

    async def batch_get(
        self, keys: list[bytes], *, model: str = "", compat_key: str = ""
    ) -> list[bytes | None]:
        """Look up N blocks in one round-trip. Returns a list of
        bytes-or-None, same length and order as ``keys`` — missing
        entries come back as None.

        Why: the per-block ``get`` path holds the connection lock for
        a full round-trip per call, serializing every GET on the
        connection. A 50K-token prompt at 16 tokens/block is ~3K
        blocks; 3K serialized GETs at ~1.5 ms each is ~5 s of pure
        UDS time. One ``batch_get`` over the same connection saves
        the per-call lock acquisition + protocol overhead, AND lets
        the kernel coalesce socket writes — measured 20-30 % overall
        speedup at this batch size, more at larger batches.

        Frame size: the daemon will inline every hit's value into the
        response frame. Practical cap is ~250 entries at ~4 MB packed
        blocks (1 GB frame). The caller is responsible for splitting
        very large requests into multiple batches if needed.

        Empty list is a no-op — returns ``[]`` without a round-trip.
        """
        if not keys:
            return []
        resp = await self._round_trip(BatchGet(keys=list(keys), model=model, compat_key=compat_key))
        if isinstance(resp, BatchGetSharedResponse):
            # Shared-arena batch — materialize each hit from the mmap.
            if (
                len(resp.offsets) != len(keys)
                or len(resp.lengths) != len(keys)
                or len(resp.versions) != len(keys)
            ):
                raise KvdProtocolError(
                    "bad_batch_get_shared_response",
                    f"got {len(resp.offsets)} offsets for {len(keys)} keys",
                )
            results: list[bytes | None] = []
            for i in range(len(keys)):
                if resp.offsets[i] < 0 or resp.lengths[i] == 0:
                    results.append(None)
                    continue
                mv = self._read_shared_slot(resp.offsets[i], resp.lengths[i], resp.versions[i])
                results.append(bytes(mv) if mv is not None else None)
            return results
        if not isinstance(resp, BatchGetResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        if len(resp.values) != len(keys):
            raise KvdProtocolError(
                "bad_batch_get_response",
                f"got {len(resp.values)} values for {len(keys)} keys",
            )
        return list(resp.values)

    async def batch_get_view(
        self, keys: list[bytes], *, model: str = "", compat_key: str = ""
    ) -> list[memoryview | bytes | None]:
        """Zero-copy variant of `batch_get`: when shared-arena is
        negotiated, returns memoryview into the local mmap directly
        (no `bytes(mv)` copy). Falls back to bytes on inline-bytes
        responses.

        Critical for the vLLM connector's hot path: 32K block loads
        per cold pass × 5 MB each would otherwise allocate ~160 GB
        of intermediate `bytes` objects via `batch_get`'s
        `bytes(mv)` materialization — defeating the whole point of
        the shared arena. Callers MUST consume the memoryview
        before another SET to the same slot, or call `bytes(mv)`
        themselves to detach.

        Empty list short-circuits with no UDS activity."""
        if not keys:
            return []
        resp = await self._round_trip(BatchGet(keys=list(keys), model=model, compat_key=compat_key))
        if isinstance(resp, BatchGetSharedResponse):
            if (
                len(resp.offsets) != len(keys)
                or len(resp.lengths) != len(keys)
                or len(resp.versions) != len(keys)
            ):
                raise KvdProtocolError(
                    "bad_batch_get_shared_response",
                    f"got {len(resp.offsets)} offsets for {len(keys)} keys",
                )
            views: list[memoryview | bytes | None] = []
            for i in range(len(keys)):
                if resp.offsets[i] < 0 or resp.lengths[i] == 0:
                    views.append(None)
                    continue
                # Zero-copy — caller owns the memoryview lifetime.
                views.append(
                    self._read_shared_slot(resp.offsets[i], resp.lengths[i], resp.versions[i])
                )
            return views
        if not isinstance(resp, BatchGetResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        if len(resp.values) != len(keys):
            raise KvdProtocolError(
                "bad_batch_get_response",
                f"got {len(resp.values)} values for {len(keys)} keys",
            )
        return list(resp.values)

    async def batch_set(
        self,
        items: list[tuple[bytes, bytes, str, dict | None]],
        *,
        model: str = "",
        compat_key: str = "",
        ttls_seconds: list[float | None] | None = None,
    ) -> list[tuple[bool, str | None]]:
        """Insert N blocks in one round-trip. Returns a parallel list
        of ``(accepted, reason)`` matching ``items`` order, mirroring
        the single-`set` return shape per element.

        Why: the per-block `set` path holds the connection lock for
        a full round-trip per call, serializing every SET. The
        connector's cold-pass save storm — one SET per packed block
        per layer for every prefill — is ~30-60s of pure UDS time
        for a 256-request bench. ``batch_set`` collapses that to
        ~1 round-trip per request flush.

        Frame-size note: the REQUEST inlines every block's value;
        practical cap is ~250 entries at 4 MB packed blobs. The
        connector chunks above that.

        ``ttls_seconds``: parallel optional TTL per item (issue #20
        item 1). When None, no TTL on any item; when a list, must
        match `items` length. Use `[None, None, 3600.0, ...]` to
        TTL only specific entries.

        Empty list is a no-op — returns ``[]`` without a round-trip.
        """
        if not items:
            return []
        if ttls_seconds is not None and len(ttls_seconds) != len(items):
            raise ValueError(
                f"ttls_seconds length {len(ttls_seconds)} != items length {len(items)}"
            )
        keys = [it[0] for it in items]
        values = [it[1] for it in items]
        retentions = [it[2] for it in items]
        metadatas = [it[3] or {} for it in items]
        resp = await self._round_trip(
            BatchSet(
                keys=keys,
                values=values,
                retentions=retentions,
                metadatas=metadatas,
                model=model,
                compat_key=compat_key,
                ttls_seconds=ttls_seconds,
            )
        )
        if not isinstance(resp, BatchSetAck):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        if len(resp.accepted) != len(items) or len(resp.reasons) != len(items):
            raise KvdProtocolError(
                "bad_batch_set_response",
                f"arity mismatch: got {len(resp.accepted)}/{len(resp.reasons)} "
                f"for {len(items)} items",
            )
        return list(zip(resp.accepted, resp.reasons, strict=True))

    async def set(
        self,
        key: bytes,
        value: bytes,
        *,
        retention: str = "short",
        model: str = "",
        compat_key: str = "",
        metadata: dict | None = None,
        ttl_seconds: float | None = None,
    ) -> tuple[bool, str | None]:
        """Returns ``(accepted, reason)``. `accepted=False` means the
        daemon refused (e.g., would displace higher-priority block);
        caller decides what to do.

        ``ttl_seconds``: optional time-to-live (issue #20 item 1).
        The daemon expires the entry lazily (on next get/exists)
        once `SET_time + ttl_seconds` has passed. Use for prompt-
        cache style scoping."""
        resp = await self._round_trip(
            Set(
                key=key,
                value=value,
                retention=retention,
                model=model,
                compat_key=compat_key,
                metadata=metadata or {},
                ttl_seconds=ttl_seconds,
            )
        )
        if not isinstance(resp, SetAck):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return resp.accepted, resp.reason

    async def set_lease(self, size: int) -> tuple[int, memoryview] | None:
        """Save-side CopyFree: ask the daemon to reserve an arena slot
        for ``size`` bytes of payload.

        Returns ``(lease_token, writable_mv)`` where ``writable_mv`` is
        a slice of the client's local arena mmap — the engine writes
        its bytes directly into this slice (GPU→host copy lands here)
        and then calls :meth:`commit_set`. Returns ``None`` when:

          - shared arena wasn't negotiated at handshake (no mmap), or
          - the server rejected the reservation (oversize, arena full
            without an evictable victim).

        The returned ``mv`` length matches ``payload_max_size`` from
        the server (= slot_size - header). The engine is free to
        write fewer bytes — the commit's ``length`` parameter tells
        the daemon how many to publish.

        Callers that fail mid-write MUST call :meth:`cancel_set` to
        release the slot; otherwise the lease stays open until the
        connection drops.
        """
        if size <= 0:
            return None
        if not self.shared_arena_negotiated or self._arena_mv is None:
            return None
        resp = await self._round_trip(SetReserve(size=size))
        if not isinstance(resp, SetReserveResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        if resp.lease_token == 0 or resp.slot_id < 0:
            logger.debug(
                "set_lease(%d) rejected by daemon: reason=%s",
                size,
                resp.reason or "<unknown>",
            )
            return None
        offset = resp.payload_offset
        end = offset + resp.payload_max_size
        # `_arena_mv` is a memoryview over the mmap; slicing returns a
        # memoryview view (zero-copy). The mapping is writable (see
        # `open_arena_view(writable=True)` in connect()).
        return resp.lease_token, self._arena_mv[offset:end]

    async def commit_set(
        self,
        lease_token: int,
        key: bytes,
        length: int,
        *,
        model: str = "",
        compat_key: str = "",
        retention: str = "default",
        ttl_seconds: int = 0,
    ) -> tuple[bool, str]:
        """Finalize a save lease. The engine has already written
        ``length`` bytes into the memoryview that :meth:`set_lease`
        handed back; we tell the daemon to publish the slot under
        ``key`` (with the usual model / compat_key / retention / TTL
        metadata, mirroring :meth:`set`).

        Returns ``(accepted, reason)``. On ``accepted=False`` the
        engine should NOT retry the same lease — the slot may have
        been released (commit failed) and the engine must obtain a
        fresh lease or fall back to inline :meth:`set`.
        """
        if length < 0:
            raise ValueError(f"commit_set length must be >= 0, got {length}")
        resp = await self._round_trip(
            SetCommit(
                lease_token=lease_token,
                key=key,
                length=length,
                model=model,
                compat_key=compat_key,
                retention=retention,
                ttl_seconds=ttl_seconds,
            )
        )
        if not isinstance(resp, SetCommitResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return resp.accepted, resp.reason

    async def cancel_set(self, lease_token: int) -> None:
        """Drop a reservation without committing. Idempotent — the
        server returns success even for unknown / already-committed
        leases, and ``wrong_owner`` is swallowed at info level here
        (no exception). Used as the safety net when a GPU→host copy
        or the surrounding flush aborts mid-write.

        Caller's exception path can call this unconditionally; the
        method never raises on protocol-level rejections.
        """
        try:
            resp = await self._round_trip(SetCancel(lease_token=lease_token))
        except KvdProtocolError as exc:
            logger.info(
                "cancel_set(%d) swallowed protocol error: %s",
                lease_token,
                exc,
            )
            return
        if not isinstance(resp, SetCancelResponse):
            # Server sent something unexpected — log and move on. The
            # whole point of cancel_set is to be the always-safe
            # exception-path hook.
            logger.info(
                "cancel_set(%d) unexpected response %s — ignoring",
                lease_token,
                type(resp).__name__,
            )

    async def exists(
        self, keys: list[bytes], *, model: str = "", compat_key: str = ""
    ) -> list[bool]:
        if not keys:
            return []
        resp = await self._round_trip(Exists(keys=list(keys), model=model, compat_key=compat_key))
        if not isinstance(resp, ExistsResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return list(resp.present)

    async def lookup_tier(
        self, key: bytes, *, model: str = "", compat_key: str = ""
    ) -> LookupTierResponse:
        """Ask the daemon which tier holds ``key`` (see wire.LookupTier).

        Protocol stub: the daemon has no LookupTier handler yet, so this
        raises ``KvdProtocolError``. Callers (e.g. the SGLang adapter's
        hipFile fast-path) wrap it in ``except (KvdConnectionError,
        KvdProtocolError)`` and degrade to the UDS Get path. Defined here
        so the method EXISTS — its previous absence raised an uncaught
        ``AttributeError`` that crashed the adapter instead of degrading —
        and so tests can monkeypatch it."""
        raise KvdProtocolError("not_implemented", "daemon has no LookupTier handler")

    async def register_file_entry(
        self,
        key: bytes,
        *,
        path: str,
        file_offset: int = 0,
        size: int = 0,
        version: int = 0,
        retention: str = "short",
        model: str = "",
        compat_key: str = "",
    ) -> tuple[bool, str]:
        """Register an on-disk chunk with the daemon (see
        wire.RegisterFileEntry) so a later LookupTier resolves it for a
        direct hipFile read.

        Protocol stub (no daemon handler yet) — raises ``KvdProtocolError``
        so the caller degrades to a UDS Set. Defined so the method exists
        (graceful fallback instead of AttributeError) and is
        monkeypatchable in tests."""
        raise KvdProtocolError("not_implemented", "daemon has no RegisterFileEntry handler")

    async def prefetch_hint(
        self,
        keys: list[bytes],
        *,
        model: str = "",
        compat_key: str = "",
        deadline_ms: int = 1000,
    ) -> None:
        """Fire-and-forget speculative L3 prefetch hint (issue #20
        item 3 / PD design §6.2). Tells the daemon to async-pull
        these block hashes from spillover/long region into the host
        RAM tier so the engine's next `get` sees a fast hit.

        Semantics differ from every other client method:
        - **No response frame** — the daemon dispatches into its
          internal worker and the client never blocks on an ack.
          We hold the connection lock just long enough to write
          the request frame.
        - **No round-trip latency** for the caller. Routers can fire
          hints in the dispatch hot path without growing TTFT.
        - **Idempotent** — keys already in host RAM are filtered by
          the daemon's worker. Over-eager hinting is cheap.
        - **Bounded by daemon-side queue** — see `--prefetch-inflight`.
          Drops on overflow are counted in `prefetch_stats` but
          invisible here.

        Empty list short-circuits with no UDS activity."""
        if not keys:
            return
        if self._reader is None or self._writer is None:
            raise KvdConnectionError("client is not connected")
        msg = PrefetchHint(
            keys=list(keys),
            model=model,
            compat_key=compat_key,
            deadline_ms=deadline_ms,
        )
        async with self._lock:
            try:
                await write_frame(self._writer, msg)
            except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as exc:
                await self._cleanup_after_error()
                raise KvdConnectionError(f"kvd prefetch_hint send failed: {exc}") from exc

    async def clear(self, *, model: str = "", compat_key: str = "") -> int:
        resp = await self._round_trip(Clear(model=model, compat_key=compat_key))
        if not isinstance(resp, ClearAck):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return resp.cleared_entries

    async def stats(self) -> StatsResponse:
        resp = await self._round_trip(Stats())
        if not isinstance(resp, StatsResponse):
            raise KvdProtocolError("unexpected_response", f"got {type(resp).__name__}")
        return resp

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read_shared_slot(self, slot_offset: int, length: int, version: int) -> memoryview | None:
        """Worker-side seqlock read from our mmap. Returns a memoryview
        into the arena on success, None on torn read (caller treats
        as miss; the value can still be fetched from kvd's long
        region on the next attempt).

        We retry once internally on torn read — version mismatch
        means the slot was overwritten between the server reading
        the version and us reading the bytes. A second attempt with
        the SAME expected_version will not succeed; the only way
        forward is to drop and let the caller re-fetch. Returning
        None here forces that.
        """
        if self._arena_mmap is None:
            # Defensive — server sent a shared response but we never
            # mmapped. Indicates a protocol bug.
            logger.warning(
                "kvd client %s: shared response without mmap — treating as miss",
                self._client_id,
            )
            return None
        return read_slot_seqlock(self._arena_mmap, slot_offset, length, version)

    def _materialize_shared(self, resp: GetSharedResponse) -> memoryview | None:
        """Single-key counterpart of the batch helper. Returns a
        memoryview or None on torn read."""
        return self._read_shared_slot(resp.slot_offset, resp.length, resp.version)

    async def _round_trip(self, msg) -> object:
        if self._reader is None or self._writer is None:
            raise KvdConnectionError("client is not connected")
        async with self._lock:
            try:
                await write_frame(self._writer, msg)
                resp = await read_frame(self._reader)
            except (asyncio.IncompleteReadError, BrokenPipeError, ConnectionResetError) as exc:
                await self._cleanup_after_error()
                raise KvdConnectionError(f"kvd round-trip failed: {exc}") from exc
        if isinstance(resp, ErrorMessage):
            raise KvdProtocolError(resp.code, resp.message)
        return resp

    async def _cleanup_after_error(self) -> None:
        if self._writer is None:
            return
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except (OSError, BrokenPipeError):
            pass
        self._reader = None
        self._writer = None
