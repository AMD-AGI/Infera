###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SCM_RIGHTS helpers — pass a file descriptor over a UDS stream socket.

We use this to hand the SharedArena's memfd from kvd to its clients
during handshake (immediately after HelloAck). The receiving process
ends up with its own FD pointing at the same kernel object, which it
can mmap independently.

The Python stdlib's `socket.sendmsg` / `recvmsg` exposes the ancillary-
data path that carries SCM_RIGHTS messages. This module wraps the
gymnastics into two helpers:

- `send_fd(sock, fd)` — writes a one-byte placeholder + the FD as
  ancillary data. The byte is necessary because POSIX says sendmsg
  must carry at least one byte of payload alongside the ancillary
  data; we use `b'F'` and the receiver discards it.

- `recv_fd(sock)` — does the matching recvmsg, returns the dup'd FD
  the kernel handed us.

Both helpers work on raw `socket.socket` objects. The asyncio variants
(`send_fd_async` / `recv_fd_async`) take an `asyncio.StreamReader` /
`StreamWriter` and route through the underlying transport's socket.
The asyncio transport's protocol doesn't expose sendmsg directly, so
we reach for `transport.get_extra_info("socket")` and call sendmsg on
it. This is fine because we're sending a tiny amount of out-of-band
data and we explicitly synchronize around it (no interleaved frames).

Wire ordering on shared-arena handshake:

  Client → Server: Hello(prefers_shared_arena=True)
  Server → Client: HelloAck(shared_arena=SharedArenaInfo(...))
  Server → Client: SCM_RIGHTS message with the arena FD

The client knows from HelloAck.shared_arena that an FD is coming and
performs the recvmsg immediately. If HelloAck.shared_arena is None
(server doesn't support, or refused), no FD message is sent.
"""

from __future__ import annotations

import array
import asyncio
import logging
import os
import socket

logger = logging.getLogger(__name__)

# One-byte payload to satisfy POSIX's "sendmsg must carry data" rule.
_SCM_RIGHTS_MARKER = b"F"

# Ancillary data buffer size — `struct.calcsize("i")` per FD, with
# a generous margin for the cmsg header. 256 bytes is plenty for
# the single-FD case.
_ANCILLARY_BUFSIZE = 256


def send_fd(sock: socket.socket, fd: int) -> None:
    """Send `fd` over `sock` via SCM_RIGHTS. The peer receives a
    DUP'd FD pointing at the same kernel object.

    `sock` must be a connected stream socket (typically `AF_UNIX,
    SOCK_STREAM` for kvd UDS). Caller owns the FD lifecycle on the
    sender side — the helper does NOT close it (the SharedArena
    keeps the FD live for its own use).
    """
    fds = array.array("i", [fd])
    ancillary = [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())]
    sock.sendmsg([_SCM_RIGHTS_MARKER], ancillary)


def recv_fd(sock: socket.socket) -> int:
    """Receive an FD via SCM_RIGHTS. Returns the FD as an int.

    Raises `RuntimeError` if no ancillary data is found in the
    received message (caller asked for an FD, server didn't send
    one — protocol bug). Raises `ConnectionError` on EOF.
    """
    marker, ancillary, flags, _addr = sock.recvmsg(len(_SCM_RIGHTS_MARKER), _ANCILLARY_BUFSIZE)
    if not marker:
        raise ConnectionError("peer closed before sending FD")
    for level, type_, data in ancillary:
        if level == socket.SOL_SOCKET and type_ == socket.SCM_RIGHTS:
            fds = array.array("i")
            # The data length must be a whole multiple of int's size;
            # truncate any trailing junk (shouldn't happen with our
            # one-FD convention but be defensive).
            n_fds = len(data) // fds.itemsize
            fds.frombytes(data[: n_fds * fds.itemsize])
            if len(fds) == 0:
                continue
            # We only send one FD per message; ignore extras (they'd
            # leak FDs in the receiver — close them to be safe).
            primary = fds[0]
            for extra in fds[1:]:
                try:
                    os.close(extra)
                except OSError:
                    pass
            return primary
    raise RuntimeError("expected SCM_RIGHTS ancillary data, got none")


# ----------------------------------------------------------------------
# asyncio-aware variants
# ----------------------------------------------------------------------


def _raw_socket_from_transport_socket(ts) -> socket.socket:
    """asyncio wraps the underlying socket in a `TransportSocket`
    that hides `sendmsg`/`recvmsg`. We need the raw socket to do
    ancillary IO; reconstruct one from the FD via `socket.socket(
    fileno=...)`, and detach it before discard so asyncio's
    transport keeps owning the FD.

    The reconstructed socket shares the FD with asyncio's transport
    (Linux dup semantics: same kernel object). Both read/write
    paths see the same byte stream. We restrict our use to
    sendmsg/recvmsg of ancillary-only messages and immediately
    detach to avoid the close-double-fd hazard.
    """
    fd = ts.fileno()
    # `fileno=fd` reuses the FD; family/type defaults to AF_UNIX,
    # SOCK_STREAM which matches asyncio.open_unix_connection.
    raw = socket.socket(fileno=fd)
    return raw


def _release_raw_socket(raw: socket.socket) -> None:
    """Detach without closing — asyncio's transport still owns the
    FD."""
    raw.detach()


async def _wait_readable_via_dup(loop: asyncio.AbstractEventLoop, ts) -> int:
    """Wait until the asyncio socket is readable. We can't add_reader
    on the same FD asyncio is using (would steal its reader callback),
    so we `dup` the FD into a separate selector slot just for the
    wakeup. The dup'd FD is closed immediately after.

    Returns the original (non-dup'd) FD so the caller can do the
    recvmsg.
    """
    real_fd = ts.fileno()
    dup_fd = os.dup(real_fd)
    fut = loop.create_future()

    def _ready():
        if not fut.done():
            fut.set_result(None)

    try:
        loop.add_reader(dup_fd, _ready)
        try:
            await fut
        finally:
            loop.remove_reader(dup_fd)
    finally:
        os.close(dup_fd)
    return real_fd


async def _wait_writable_via_dup(loop: asyncio.AbstractEventLoop, ts) -> int:
    """Counterpart of `_wait_readable_via_dup` for sendmsg."""
    real_fd = ts.fileno()
    dup_fd = os.dup(real_fd)
    fut = loop.create_future()

    def _ready():
        if not fut.done():
            fut.set_result(None)

    try:
        loop.add_writer(dup_fd, _ready)
        try:
            await fut
        finally:
            loop.remove_writer(dup_fd)
    finally:
        os.close(dup_fd)
    return real_fd


async def _recv_fd_async_inner(ts) -> int:
    """Non-blocking recv_fd that waits for readable via the event
    loop on a dup'd FD (so we don't steal asyncio's reader callback
    on the real FD)."""
    loop = asyncio.get_running_loop()
    while True:
        raw = _raw_socket_from_transport_socket(ts)
        try:
            try:
                return recv_fd(raw)
            except BlockingIOError:
                pass
        finally:
            _release_raw_socket(raw)
        await _wait_readable_via_dup(loop, ts)


async def _send_fd_async_inner(ts, fd: int) -> None:
    """Non-blocking send_fd. Tiny ancillary message, rarely blocks
    in practice (send buffer empty after writer.drain), but we
    handle EAGAIN cleanly."""
    loop = asyncio.get_running_loop()
    while True:
        raw = _raw_socket_from_transport_socket(ts)
        try:
            try:
                send_fd(raw, fd)
                return
            except BlockingIOError:
                pass
        finally:
            _release_raw_socket(raw)
        await _wait_writable_via_dup(loop, ts)


async def send_fd_async(writer: asyncio.StreamWriter, fd: int) -> None:
    """Drain `writer`, then send `fd` via SCM_RIGHTS on the underlying
    socket. The drain ensures any framed data already written by
    `write_frame` has been pushed before the ancillary message —
    otherwise the receiver might pull the FD ancillary off a recvmsg
    that should be servicing the framed bytes.

    Note: SCM_RIGHTS messages are sent on the same byte stream but
    are received via `recvmsg`, not `recv`. The kernel matches
    ancillary data to the FIRST byte received in that recvmsg call.
    Our protocol enforces strict ordering: server `write_frame(HelloAck)`
    then `await drain()` then send the FD; client `read_frame` returns
    after `readexactly` over framed bytes, then calls `recv_fd` which
    does a single `recvmsg(1, ...)`.
    """
    await writer.drain()
    ts = writer.get_extra_info("socket")
    if ts is None:
        raise RuntimeError("writer has no underlying socket — can't send FD")
    await _send_fd_async_inner(ts, fd)


async def recv_fd_async(reader: asyncio.StreamReader) -> int:
    """Receive an FD via SCM_RIGHTS on the underlying socket of
    `reader`. The reader's transport must be a stream socket
    (i.e. opened via `asyncio.open_unix_connection`).

    **Important**: this MUST be called when no data is pending in
    `reader`'s buffer — otherwise asyncio's transport will have
    already pumped the FD ancillary off the socket (and dropped it,
    since the transport's read path doesn't capture ancillary data).

    Our wire protocol guarantees this: the server writes HelloAck,
    drains, then sends the FD. The client reads HelloAck (a complete
    framed message) via `read_frame`, which leaves the socket buffer
    empty, and then calls `recv_fd_async` BEFORE issuing any other
    read. We pause the transport's reader during the recvmsg to
    prevent it from racing with our call.
    """
    transport = reader._transport  # private but stable across asyncio versions
    if transport is None:
        raise RuntimeError("reader has no transport")
    ts = transport.get_extra_info("socket")
    if ts is None:
        raise RuntimeError("transport has no socket — can't recv FD")

    # Pause asyncio's reader so it doesn't race us on recvmsg. Stops
    # the transport from calling sock.recv() which would consume the
    # ancillary message (and the kernel would NOT redeliver it to
    # us). `pause_reading` is documented and safe on all stream
    # transports.
    transport.pause_reading()
    try:
        return await _recv_fd_async_inner(ts)
    finally:
        transport.resume_reading()


__all__ = [
    "send_fd",
    "recv_fd",
    "send_fd_async",
    "recv_fd_async",
]
