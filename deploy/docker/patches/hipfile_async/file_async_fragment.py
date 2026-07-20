###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
# --- BEGIN hipfile-async-patch additions ---
# Appended by deploy/docker/patches/hipfile_async/patch_hipfile_async.sh.
# Adds Stream context manager + supports_async() probe + write_async /
# read_async methods on FileHandle. Mirrors the cuFile async pattern.
#
# CRITICAL DESIGN POINT (root cause of HipFileException 5022 in early
# Cython-based wrappers): hipFileReadAsync / hipFileWriteAsync take
# pointer args (size_p, file_off_p, buf_off_p, bytes_done_p) that the
# driver dereferences AFTER the C function returns (the actual IO
# completes when the stream is synchronised). Cython `cdef`
# stack-locals passed as `&local` go out of scope when the wrapper
# returns — driver then writes `bytes_done` to a dead address and may
# re-read the size pointer when chunking the IO. The race shows up as
# (a) `bytes_done` always reading back as 0, and (b) occasional 5022
# "invalid value" errors on subsequent calls (the stack address got
# reused with incompatible bytes mid-submission).
#
# Fix: call libhipfile.so directly via ctypes with HEAP-allocated
# c_size_t / c_int64 / c_ssize_t slots wrapped in a Python object
# (_AsyncIOHandle) returned to the caller. Caller MUST keep the handle
# alive past stream.synchronize() — easiest pattern: stash it
# alongside the cuda.Event that gates the consumer's downstream work.

import ctypes

from hipfile._hipfile import (
    hipFileStreamDeregister as _hipFileStreamDeregister,
)

# Cython-wrapped stream register/deregister + async-support probe.
# These wrap pure scalar arguments, so the Cython stack-locals issue
# that affects read/write_async doesn't apply here — keep using the
# binding for stream lifecycle + capability probe.
from hipfile._hipfile import (  # noqa: E402
    hipFileStreamRegister as _hipFileStreamRegister,
)
from hipfile._hipfile import (
    supports_async as _supports_async,
)

_lib = ctypes.CDLL("libhipfile.so", mode=ctypes.RTLD_GLOBAL)


class _hipFileError_t_struct(ctypes.Structure):
    """Mirror of the C ``hipFileError_t`` struct (see hipfile.h).
    `err` is the high-level hipFileOpError_t code (0 = success);
    `hip_drv_err` is the underlying hipError_t when err ==
    hipFileHipDriverError."""

    _fields_ = [("err", ctypes.c_int), ("hip_drv_err", ctypes.c_int)]


_lib.hipFileReadAsync.restype = _hipFileError_t_struct
_lib.hipFileReadAsync.argtypes = [
    ctypes.c_void_p,  # hipFileHandle_t
    ctypes.c_void_p,  # buffer_base (device ptr)
    ctypes.POINTER(ctypes.c_size_t),  # size_p (in/out)
    ctypes.POINTER(ctypes.c_int64),  # file_offset_p (in/out)
    ctypes.POINTER(ctypes.c_int64),  # buffer_offset_p (in/out)
    ctypes.POINTER(ctypes.c_ssize_t),  # bytes_done_p (out)
    ctypes.c_void_p,  # hipStream_t
]
_lib.hipFileWriteAsync.restype = _hipFileError_t_struct
_lib.hipFileWriteAsync.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_size_t),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_int64),
    ctypes.POINTER(ctypes.c_ssize_t),
    ctypes.c_void_p,
]


class _AsyncIOHandle:
    """Owns the four in/out slots for the lifetime of an async submit.

    Caller MUST keep the handle alive until ``stream.synchronize()``
    (or equivalent wait on the cuda.Event the stream produced)
    returns. Then read ``handle.bytes_done`` for the byte count.

    Easiest pattern: store the handle in the same dict the caller
    uses to track the per-layer ``cuda.Event`` — the Event lifetime
    already matches the I/O lifetime exactly.

    Slots are ``ctypes.c_*`` values; ``ctypes.byref(slot)`` gives
    a stable address that survives until the Python ``_AsyncIOHandle``
    is garbage-collected.
    """

    __slots__ = ("size", "file_off", "buf_off", "_bytes_done")

    def __init__(self, size: int, file_off: int, buf_off: int):
        self.size = ctypes.c_size_t(int(size))
        self.file_off = ctypes.c_int64(int(file_off))
        self.buf_off = ctypes.c_int64(int(buf_off))
        self._bytes_done = ctypes.c_ssize_t(0)

    @property
    def bytes_done(self) -> int:
        """The driver updates this field once the async I/O completes
        on the stream. Read AFTER stream.synchronize / event.wait."""
        return int(self._bytes_done.value)


def _hipfile_async_install_filehandle_methods():
    """Attach write_async / read_async to FileHandle. Idempotent."""

    def write_async(
        self,
        src,
        size: int,
        file_offset: int,
        src_offset: int,
        stream_handle: int,
    ) -> "_AsyncIOHandle":
        """Submit an async write to ``stream_handle``. Returns an
        ``_AsyncIOHandle`` the caller MUST keep alive past
        ``stream.synchronize()``. After sync, ``handle.bytes_done``
        is the actual bytes written.

        On submit failure raises ``HipFileException``. The driver may
        still process previously-queued IOs on the stream after a
        submit failure — caller should treat the stream as poisoned
        if multiple submits fail in a row.
        """
        if self._handle is None:
            raise RuntimeError("HipFile.write_async called on a closed handle")
        src_ptr = src.ptr if hasattr(src, "ptr") else int(src)
        h = _AsyncIOHandle(size, file_offset, src_offset)
        err = _lib.hipFileWriteAsync(
            ctypes.c_void_p(int(self._handle)),
            ctypes.c_void_p(int(src_ptr)),
            ctypes.byref(h.size),
            ctypes.byref(h.file_off),
            ctypes.byref(h.buf_off),
            ctypes.byref(h._bytes_done),
            ctypes.c_void_p(int(stream_handle)),
        )
        if err.err != 0:
            raise HipFileException(err.err, err.hip_drv_err)
        return h

    def read_async(
        self,
        dest,
        size: int,
        file_offset: int,
        dest_offset: int,
        stream_handle: int,
    ) -> "_AsyncIOHandle":
        """Mirror of ``write_async`` for reads. Returns an
        ``_AsyncIOHandle`` — keep alive past stream sync, then
        read ``handle.bytes_done``."""
        if self._handle is None:
            raise RuntimeError("HipFile.read_async called on a closed handle")
        dest_ptr = dest.ptr if hasattr(dest, "ptr") else int(dest)
        h = _AsyncIOHandle(size, file_offset, dest_offset)
        err = _lib.hipFileReadAsync(
            ctypes.c_void_p(int(self._handle)),
            ctypes.c_void_p(int(dest_ptr)),
            ctypes.byref(h.size),
            ctypes.byref(h.file_off),
            ctypes.byref(h.buf_off),
            ctypes.byref(h._bytes_done),
            ctypes.c_void_p(int(stream_handle)),
        )
        if err.err != 0:
            raise HipFileException(err.err, err.hip_drv_err)
        return h

    if not hasattr(FileHandle, "write_async"):
        FileHandle.write_async = write_async  # type: ignore[attr-defined]
    if not hasattr(FileHandle, "read_async"):
        FileHandle.read_async = read_async  # type: ignore[attr-defined]


_hipfile_async_install_filehandle_methods()


class Stream:
    """Context-managed registration of a CUDA/HIP stream with hipFile.

    Wraps the integer stream handle (``torch.cuda.Stream.cuda_stream``)
    so async I/O calls can target it. Usage::

        s = torch.cuda.Stream(device=device)
        with Stream(s.cuda_stream) as st:
            handle = fh.read_async(buf, size, file_off, buf_off, st.handle)
            event = torch.cuda.Event()
            event.record(s)
            default_stream.wait_event(event)
            # ... default_stream consumer reads buf safely now ...
            # Caller MUST keep `handle` alive until after the wait_event
            # (downstream consumer is the sync point).
    """

    def __init__(self, stream_handle: int, flags: int = 0):
        self._stream = int(stream_handle)
        self._flags = int(flags)
        self._registered = False

    @property
    def handle(self) -> int:
        return self._stream

    def register(self) -> None:
        if self._registered:
            return
        rc = _hipFileStreamRegister(self._stream, self._flags)
        if rc != 0:
            raise HipFileException(rc, 0)
        self._registered = True

    def deregister(self) -> None:
        if not self._registered:
            return
        rc = _hipFileStreamDeregister(self._stream)
        self._registered = False
        if rc != 0:
            raise HipFileException(rc, 0)

    def __enter__(self) -> "Stream":
        self.register()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.deregister()


def supports_async() -> bool:
    """Returns True if the loaded ``libhipfile.so`` supports the async
    API (binding wrapper + driver both wired). False indicates the
    driver is sync-only — caller should fall back to the sync write/
    read path."""
    return bool(_supports_async())


# --- END hipfile-async-patch additions ---
