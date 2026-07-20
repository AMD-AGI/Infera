###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for `infera.kvd.fd_passing` — SCM_RIGHTS over UDS.

Validates that:
- a memfd written through one end of a UDS pair appears at the other
  end as a fresh FD pointing at the same kernel object
- both ends, when they mmap the FD, see each other's writes
"""

from __future__ import annotations

import asyncio
import mmap
import os
import socket

import pytest

from infera.kvd.fd_passing import (
    recv_fd,
    recv_fd_async,
    send_fd,
    send_fd_async,
)


def test_send_recv_fd_over_socketpair():
    """Open a memfd, send it through one end of a UDS socketpair,
    receive it on the other, and verify both ends see the same
    bytes through their own mmap."""
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        # Create a memfd and write a marker.
        fd = os.memfd_create("test-shared", os.MFD_CLOEXEC)
        try:
            os.ftruncate(fd, 4096)
            os.lseek(fd, 0, 0)
            os.write(fd, b"hello from sender" + b"\x00" * 100)

            # Send through `parent`.
            send_fd(parent, fd)

            # Receive on `child`.
            received_fd = recv_fd(child)
            try:
                assert received_fd > 0
                # The received FD must point at the same kernel object.
                # Verify by reading bytes back through the received FD.
                os.lseek(received_fd, 0, 0)
                data = os.read(received_fd, 17)
                assert data == b"hello from sender"

                # And vice versa — write through child FD, read through
                # parent FD.
                os.lseek(received_fd, 100, 0)
                os.write(received_fd, b"from receiver")
                os.lseek(fd, 100, 0)
                assert os.read(fd, 13) == b"from receiver"
            finally:
                os.close(received_fd)
        finally:
            os.close(fd)
    finally:
        parent.close()
        child.close()


def test_recv_fd_raises_on_eof():
    """If the sender closes without sending an FD, `recv_fd` raises
    `ConnectionError` (the empty marker byte signals close)."""
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        parent.close()
        with pytest.raises(ConnectionError):
            recv_fd(child)
    finally:
        try:
            child.close()
        except OSError:
            pass


def test_mmap_shared_across_send_recv():
    """The whole point of SCM_RIGHTS for our use case: both processes
    can mmap the same FD and see each other's writes. We simulate
    the cross-process scenario in-process via two FDs that the
    kernel guarantees point at the same memfd object."""
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        fd = os.memfd_create("shared-arena", os.MFD_CLOEXEC)
        os.ftruncate(fd, 4096)
        send_fd(parent, fd)
        received_fd = recv_fd(child)

        # Both FDs → mmap the same 4 KiB region.
        try:
            sender_mm = mmap.mmap(
                fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )
            receiver_mm = mmap.mmap(
                received_fd, 4096, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ | mmap.PROT_WRITE
            )

            try:
                # Sender writes.
                sender_mm[0:11] = b"hello world"
                # Receiver sees.
                assert bytes(receiver_mm[0:11]) == b"hello world"
                # Receiver writes.
                receiver_mm[100:108] = b"response"
                assert bytes(sender_mm[100:108]) == b"response"
            finally:
                sender_mm.close()
                receiver_mm.close()
        finally:
            os.close(fd)
            os.close(received_fd)
    finally:
        parent.close()
        child.close()


# ----------------------------------------------------------------------
# asyncio variants — sanity check the helper plumbing works against an
# asyncio.StreamReader/Writer pair.
# ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_send_recv_fd_async():
    """Open a UDS pair with asyncio, exchange an FD via the async
    helpers."""
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        reader_a, writer_a = await asyncio.open_unix_connection(sock=a)
        reader_b, writer_b = await asyncio.open_unix_connection(sock=b)
        try:
            fd = os.memfd_create("test-async", os.MFD_CLOEXEC)
            try:
                os.ftruncate(fd, 1024)
                os.lseek(fd, 0, 0)
                os.write(fd, b"async-payload")

                # Send via async helper on writer_a.
                send_task = asyncio.create_task(send_fd_async(writer_a, fd))
                received_fd = await recv_fd_async(reader_b)
                await send_task

                try:
                    os.lseek(received_fd, 0, 0)
                    assert os.read(received_fd, 13) == b"async-payload"
                finally:
                    os.close(received_fd)
            finally:
                os.close(fd)
        finally:
            writer_a.close()
            await writer_a.wait_closed()
            writer_b.close()
            await writer_b.wait_closed()
    except Exception:
        # The sockets are owned by asyncio after open_unix_connection
        # so we don't double-close here.
        raise
