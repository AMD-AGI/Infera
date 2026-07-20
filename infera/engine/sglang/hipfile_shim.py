###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Thin Python wrapper around the ROCm hipFile native binding.

Purpose
-------
hipFile is AMD's GPU-direct storage API (the cuFile counterpart on
ROCm). The official Python binding lives in
``ROCm/hipFile`` / ``rocm-systems/projects/hipfile`` and is built
out-of-tree by ``deploy/docker/scripts/build_hipfile.sh`` (task #12) — there is no
PyPI wheel. This module isolates that binding behind a narrow Python
surface that ``infera/engine/sglang/kvd_adapter.py`` can use from
the L3 / NVMe read path (task #18, Phase F1) without leaking
``hipfile.*`` types through the rest of the codebase.

Design notes
------------
- The binding (``import hipfile``) is loaded lazily inside method
  bodies and inside :func:`is_available`. The shim therefore imports
  cleanly on hosts without ROCm or hipFile installed; the adapter
  uses :func:`is_available` to gate the GPU-direct path and falls back
  to the POSIX path otherwise.

- ``HipFileDriver`` is a process-singleton. The native ``Driver``
  carries IOMMU mappings and per-process state — instantiating it
  twice produces undefined behavior. We keep one driver per process,
  opened on first ``ensure_open()`` and torn down via ``close()``
  (or interpreter shutdown).

- ``RegisteredBuffer`` wraps the ``hipfile.Buffer(ptr, size, 0)``
  ``.register()`` / ``.deregister()`` lifecycle. The registered region
  is the SGLang ``HostKVCache`` buffer. The adapter
  builds two of these for GLM-5.1 DSA + NSA — one over
  ``MLATokenToKVPoolHost.kv_buffer``, one over
  ``NSAIndexerPoolHost.index_k_with_scale_buffer``.

- ``HipFile`` wraps ``hipfile.FileHandle`` as a context manager with
  a Python-friendly mode string ("r" / "w" / "r+") and synchronous
  ``read`` / ``write`` taking integer device pointers (matches what
  ``ctypes.c_void_p.value`` / ``Tensor.data_ptr()`` produce).

This file is original code. The LMCache project's similar shim was
consulted for the underlying hipFile API shape only — no code is
copied, no LMCache attribution is owed.

API surface kept deliberately narrow: only what task #18 needs at F1.
Extend in a follow-up PR, not speculatively.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover — typing only
    from types import ModuleType


logger = logging.getLogger(__name__)


# 4 KiB is the alignment hipFile (and cuFile) require for both file
# offsets and buffer pointers when O_DIRECT-style DMA is used. Smaller
# alignments are accepted only when the driver falls back to its
# compat path — exactly the silent-degradation mode we want to avoid
# (silent compat-mode). Enforce up front.
HIPFILE_ALIGNMENT = 4096

_INSTALL_HINT = (
    "hipfile python package is not installed. Build it on this host with "
    "deploy/docker/scripts/build_hipfile.sh (see task #12) — there is no PyPI wheel. The "
    "infera GPU-direct L3 path will fall back to POSIX without it."
)


def _import_hipfile() -> ModuleType:
    """Lazy import of the native ``hipfile`` binding.

    Raises ``ImportError`` with an actionable hint when missing — callers
    that should *not* hard-fail (e.g. :func:`is_available`) catch this.
    """
    try:
        import hipfile  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — exercised in tests via mocking
        raise ImportError(_INSTALL_HINT) from exc
    return hipfile


# ----------------------------------------------------------------------
# Driver — process singleton
# ----------------------------------------------------------------------


class HipFileDriver:
    """Process-singleton wrapper over ``hipfile.Driver``.

    The native ``Driver`` opens kernel state (IOMMU / BAR mappings)
    that is one-per-process; instantiating it twice is undefined.
    Construct this class freely — the second and later instances share
    the underlying driver. ``close()`` is intended for explicit
    teardown (tests, daemon restart). Interpreter shutdown also
    triggers ``__del__`` on the singleton instance.
    """

    _singleton_lock = threading.Lock()
    _singleton: HipFileDriver | None = None

    def __new__(cls) -> HipFileDriver:
        with cls._singleton_lock:
            if cls._singleton is None:
                instance = super().__new__(cls)
                instance._driver = None  # type: ignore[attr-defined]
                instance._opened = False  # type: ignore[attr-defined]
                instance._instance_lock = threading.Lock()  # type: ignore[attr-defined]
                cls._singleton = instance
        return cls._singleton

    # Attribute hints for the type checker — actually set in __new__.
    _driver: object | None
    _opened: bool
    _instance_lock: threading.Lock

    def ensure_open(self) -> object:
        """Open the underlying driver if not already open. Returns the
        native ``hipfile.Driver`` instance (callers normally don't need
        it — :class:`HipFile` and :class:`RegisteredBuffer` reach
        through the singleton transparently).

        Raises ``ImportError`` (from :func:`_import_hipfile`) if the
        binding isn't installed.
        """
        with self._instance_lock:
            if not self._opened:
                hipfile = _import_hipfile()
                self._driver = hipfile.Driver()
                self._opened = True
                logger.info("hipfile driver opened (pid=%d)", os.getpid())
            return self._driver  # type: ignore[return-value]

    def is_open(self) -> bool:
        return self._opened

    def close(self) -> None:
        """Explicit teardown. Idempotent. After close, ``ensure_open()``
        will re-open. Mostly used by tests."""
        with self._instance_lock:
            if not self._opened:
                return
            driver = self._driver
            self._driver = None
            self._opened = False
        # The native Driver doesn't expose an explicit close — its
        # __del__ does the work. Drop our reference; GC handles the
        # rest. We swallow exceptions: teardown should never raise.
        try:
            del driver
        except Exception:  # pragma: no cover — defensive
            logger.debug("hipfile driver del raised; ignoring", exc_info=True)

    def __del__(self) -> None:  # pragma: no cover — interpreter shutdown
        try:
            self.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Buffer registration — used at SGLang HostKVCache init time
# ----------------------------------------------------------------------


@dataclass
class _BufferSpec:
    """Snapshot of the register() arguments — handy for diagnostics."""

    ptr: int
    size: int
    flags: int


class RegisteredBuffer:
    """Context-manager wrapping ``hipfile.Buffer(ptr, size, flags).register()``.

    The bounded region (``ptr``, ``size``) must be a host-pinned buffer
    (``cudaHostRegister`` or equivalent). After ``__enter__`` the region is
    available as a hipFile DMA target until ``__exit__``.

    Validates:
      - ``ptr != 0``
      - ``size > 0``
      - 4 KiB alignment on both ``ptr`` and ``size``

    Misalignment is rejected loudly rather than letting hipFile fall
    back to its compat (CPU-bounce) path, which would silently destroy
    the throughput win the GPU-direct backend exists for.
    """

    def __init__(self, ptr: int, size: int, flags: int = 0) -> None:
        if ptr == 0:
            raise ValueError("RegisteredBuffer: ptr must be non-NULL")
        if size <= 0:
            raise ValueError(f"RegisteredBuffer: size must be > 0 (got {size})")
        if ptr % HIPFILE_ALIGNMENT != 0:
            raise ValueError(
                f"RegisteredBuffer: ptr 0x{ptr:x} is not "
                f"{HIPFILE_ALIGNMENT}-byte aligned (hipFile DMA requirement)"
            )
        if size % HIPFILE_ALIGNMENT != 0:
            raise ValueError(
                f"RegisteredBuffer: size {size} is not a multiple of "
                f"{HIPFILE_ALIGNMENT} (hipFile DMA requirement)"
            )
        self._spec = _BufferSpec(ptr=ptr, size=size, flags=flags)
        self._buffer: object | None = None
        self._registered = False

    def __enter__(self) -> RegisteredBuffer:
        # Ensure the driver is open before we register against it.
        HipFileDriver().ensure_open()
        hipfile = _import_hipfile()
        self._buffer = hipfile.Buffer(self._spec.ptr, self._spec.size, self._spec.flags)
        self._buffer.register()  # type: ignore[union-attr]
        self._registered = True
        logger.debug(
            "registered hipfile buffer ptr=0x%x size=%d flags=%d",
            self._spec.ptr,
            self._spec.size,
            self._spec.flags,
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.deregister()

    def deregister(self) -> None:
        """Idempotent. Safe to call from ``__del__`` as well as
        ``__exit__``."""
        if not self._registered:
            return
        try:
            if self._buffer is not None:
                self._buffer.deregister()  # type: ignore[union-attr]
        except Exception:  # pragma: no cover — defensive
            logger.warning(
                "hipfile buffer deregister raised for ptr=0x%x size=%d",
                self._spec.ptr,
                self._spec.size,
                exc_info=True,
            )
        finally:
            self._registered = False
            self._buffer = None

    @property
    def handle(self) -> object | None:
        """Underlying `hipfile.Buffer` instance, exposed for callers
        that need to pass it to the FileHandle read/write APIs (the
        current binding wants a `Buffer` + `buffer_offset` rather
        than a raw device pointer). `None` outside the registered
        window."""
        return self._buffer

    @property
    def base_ptr(self) -> int:
        """The aligned device pointer the buffer was registered with.
        Useful for callers that compute `buffer_offset = slot_ptr -
        base_ptr` when they only know slot_ptr directly."""
        return self._spec.ptr

    def __del__(self) -> None:  # pragma: no cover — interpreter timing
        try:
            self.deregister()
        except Exception:
            pass


# ----------------------------------------------------------------------
# File handle — context manager wrapping hipfile.FileHandle
# ----------------------------------------------------------------------


# Mapping from our Python-style mode strings to (open() flags, hipfile
# direction descriptor). hipFile needs O_DIRECT-friendly flags; we
# layer O_DIRECT on top when ``direct=True`` is requested.
_MODE_TO_FLAGS = {
    "r": os.O_RDONLY,
    "w": os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    "r+": os.O_RDWR,
}


class HipFile:
    """Context manager wrapping a POSIX fd + ``hipfile.FileHandle``.

    Usage::

        with HipFile("/path/to/blob", "r") as f:
            n = f.read(dest_ptr, size, file_offset=offset)

    ``dest_ptr`` / ``src_ptr`` are integer device pointers — typically
    obtained from ``ctypes.c_void_p.value`` or ``torch.Tensor.data_ptr()``
    after the underlying buffer has been registered via
    :class:`RegisteredBuffer`.

    ``mode`` is "r", "w", or "r+". The POSIX file is opened with
    matching flags; ``direct=True`` adds ``O_DIRECT`` for the
    DMA-friendly path (the alignment guards on
    :class:`RegisteredBuffer` already enforce the buffer side; the
    caller is responsible for aligned file offsets when ``direct``).
    """

    def __init__(self, path: str, mode: str = "r", *, direct: bool = False) -> None:
        if mode not in _MODE_TO_FLAGS:
            raise ValueError(
                f"HipFile: unsupported mode {mode!r} (use one of {list(_MODE_TO_FLAGS)})"
            )
        self._path = path
        self._mode = mode
        self._direct = direct
        self._fd: int | None = None
        self._handle: object | None = None

    def __enter__(self) -> HipFile:
        HipFileDriver().ensure_open()
        hipfile = _import_hipfile()
        flags = _MODE_TO_FLAGS[self._mode]
        if self._direct:
            flags |= getattr(os, "O_DIRECT", 0)

        # New binding API (rocm-systems/hipfile current):
        # `FileHandle(path, flags, mode=0o644)` constructs but does NOT
        # open. Call `.open()` (or use the binding's context manager)
        # to actually open the fd inside the binding — it owns the fd
        # lifecycle; we don't `os.open` it ourselves.
        #
        # Legacy binding (older builds): `FileHandle(fd)` took an
        # already-opened fd. We auto-detect via TypeError on the new
        # signature and fall back to the legacy path so this shim
        # tolerates both binding versions installed across CI / lab
        # hosts. The legacy path retains the original fd-cleanup
        # behavior.
        self._fd = None
        try:
            self._handle = hipfile.FileHandle(self._path, flags, 0o644)
        except TypeError:
            # Legacy binding — open the fd here and pass it in.
            self._fd = os.open(self._path, flags, 0o644)
            try:
                self._handle = hipfile.FileHandle(self._fd)
            except Exception:
                try:
                    os.close(self._fd)
                finally:
                    self._fd = None
                raise
        else:
            # New binding — open the handle now (binding owns the fd).
            open_fn = getattr(self._handle, "open", None)
            if callable(open_fn):
                try:
                    open_fn()
                except Exception:
                    # Best-effort handle close if open failed.
                    try:
                        close_fn = getattr(self._handle, "close", None)
                        if callable(close_fn):
                            close_fn()
                    except Exception:  # pragma: no cover — defensive
                        pass
                    self._handle = None
                    raise
        logger.debug(
            "hipfile opened path=%s mode=%s (legacy_fd=%s)",
            self._path,
            self._mode,
            "yes" if self._fd is not None else "no",
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        """Idempotent."""
        handle = self._handle
        fd = self._fd
        self._handle = None
        self._fd = None
        if handle is not None:
            try:
                # hipFile's FileHandle close path is via deletion; some
                # builds expose an explicit `close()` — call it if present
                # so the kernel-side state is released before fd close.
                close_fn = getattr(handle, "close", None)
                if callable(close_fn):
                    close_fn()
            except Exception:  # pragma: no cover — defensive
                logger.debug("hipfile FileHandle.close raised; ignoring", exc_info=True)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:  # pragma: no cover — defensive
                logger.debug("os.close raised for hipfile fd=%d; ignoring", fd, exc_info=True)

    def read(
        self,
        dest: int | object,
        size: int,
        file_offset: int = 0,
        dest_offset: int = 0,
    ) -> int:
        """Read ``size`` bytes from ``file_offset`` into the registered
        device buffer at ``dest_offset`` within the buffer.

        ``dest`` is either:
          - a ``hipfile.Buffer`` object (new binding, current
            rocm-systems master) — pass the underlying buffer of a
            ``RegisteredBuffer`` via ``RegisteredBuffer.handle``.
          - an integer device pointer (legacy binding) — value of
            ``ptr.value`` or ``tensor.data_ptr()`` for the target slot
            inside a previously-registered region.

        ``dest_offset`` is the byte offset within the Buffer (new
        binding); with the legacy binding it's added to ``dest_ptr``
        before the read. Both forms are byte-addressable, so callers
        can mix-and-match a single addressing scheme.

        Returns the number of bytes transferred. Caller checks against
        ``size`` if a short read is unexpected (POSIX-pread semantics
        on a non-fatal short read).
        """
        if self._handle is None:
            raise RuntimeError("HipFile.read called on a closed handle")
        return int(self._handle.read(dest, size, file_offset, dest_offset))  # type: ignore[union-attr]

    def write(
        self,
        src: int | object,
        size: int,
        file_offset: int = 0,
        src_offset: int = 0,
    ) -> int:
        """Write ``size`` bytes from the registered device buffer at
        ``src_offset`` to the file at ``file_offset``. ``src`` accepts
        the same forms as :meth:`read`'s ``dest`` (Buffer or int)."""
        if self._handle is None:
            raise RuntimeError("HipFile.write called on a closed handle")
        return int(self._handle.write(src, size, file_offset, src_offset))  # type: ignore[union-attr]


# ----------------------------------------------------------------------
# Availability probe
# ----------------------------------------------------------------------


def is_available() -> bool:
    """Return True if the hipFile binding can be imported AND the
    process-singleton driver opens. Used by ``kvd_adapter`` to gate
    the GPU-direct read/write path; on False, the adapter falls back
    to the POSIX path.

    Logs a single WARNING describing why on each failure mode, so
    operators can see whether the missing piece is the binding (build
    deploy/docker/scripts/build_hipfile.sh) or the driver (kernel module / IOMMU).
    """
    try:
        _import_hipfile()
    except ImportError as exc:
        logger.warning("hipfile binding unavailable: %s", exc)
        return False
    try:
        HipFileDriver().ensure_open()
    except Exception as exc:  # pragma: no cover — exercised only on broken hosts
        logger.warning("hipfile driver failed to open: %s", exc)
        return False
    return True
