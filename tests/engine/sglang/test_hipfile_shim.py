###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/engine/sglang/hipfile_shim.py.

The shim is a thin Python wrapper around the native ``hipfile`` binding
(ROCm's GPU-direct storage API counterpart to cuFile). The binding
itself is built out-of-tree by deploy/docker/scripts/build_hipfile.sh and is NOT
available on a CPU-only dev box.

These tests therefore install a fake ``hipfile`` module via
``sys.modules`` injection BEFORE the shim's lazy importer runs
(``_import_hipfile`` calls ``import hipfile`` on every entry — no
module-level cache to invalidate). The fake records every call so we
can assert lifecycle correctness end-to-end without touching real GPU.

Mirrors the style of test_sglang_kvd_adapter.py.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types

import pytest

from infera.engine.sglang.hipfile_shim import (
    HIPFILE_ALIGNMENT,
    HipFile,
    HipFileDriver,
    RegisteredBuffer,
    is_available,
)

# ----------------------------------------------------------------------
# Fake hipfile binding
# ----------------------------------------------------------------------


class _FakeDriver:
    """Records construction; the shim only stores the instance."""

    instances: list[_FakeDriver] = []

    def __init__(self) -> None:
        _FakeDriver.instances.append(self)


class _FakeBuffer:
    """Records register/deregister calls + constructor args."""

    constructed: list[tuple[int, int, int]] = []

    def __init__(self, ptr: int, size: int, flags: int) -> None:
        self.ptr = ptr
        self.size = size
        self.flags = flags
        self.register_calls = 0
        self.deregister_calls = 0
        _FakeBuffer.constructed.append((ptr, size, flags))

    def register(self) -> None:
        self.register_calls += 1

    def deregister(self) -> None:
        self.deregister_calls += 1


class _FakeFileHandle:
    """Records open (constructor), close, read, write."""

    constructed: list[_FakeFileHandle] = []

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.closed = False
        self.read_calls: list[tuple[int, int, int, int]] = []
        self.write_calls: list[tuple[int, int, int, int]] = []
        # Bytes the next read/write call will report it transferred.
        self.next_return = 0
        _FakeFileHandle.constructed.append(self)

    def close(self) -> None:
        self.closed = True

    def read(self, dest_ptr: int, size: int, file_offset: int, dest_offset: int) -> int:
        self.read_calls.append((dest_ptr, size, file_offset, dest_offset))
        return self.next_return

    def write(self, src_ptr: int, size: int, file_offset: int, src_offset: int) -> int:
        self.write_calls.append((src_ptr, size, file_offset, src_offset))
        return self.next_return


def _make_fake_hipfile_module() -> types.ModuleType:
    mod = types.ModuleType("hipfile")
    mod.Driver = _FakeDriver
    mod.Buffer = _FakeBuffer
    mod.FileHandle = _FakeFileHandle
    return mod


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_driver_singleton():
    """Each test gets a fresh HipFileDriver singleton.

    The shim stores the singleton on the class (``HipFileDriver._singleton``);
    without this reset, state from one test (e.g. an opened driver) leaks
    into the next test's ``is_available()`` probe.
    """
    HipFileDriver._singleton = None
    _FakeDriver.instances.clear()
    _FakeBuffer.constructed.clear()
    _FakeFileHandle.constructed.clear()
    yield
    HipFileDriver._singleton = None


@pytest.fixture
def fake_hipfile(monkeypatch):
    """Inject a fake ``hipfile`` binding for the shim's lazy importer."""
    mod = _make_fake_hipfile_module()
    monkeypatch.setitem(sys.modules, "hipfile", mod)
    yield mod


@pytest.fixture
def missing_hipfile(monkeypatch):
    """Ensure the lazy importer raises ImportError (no fake installed)."""
    # Use raising=False because the binding genuinely isn't installed on
    # the CI box; this just guards against test ordering where a prior
    # test left a fake behind.
    monkeypatch.delitem(sys.modules, "hipfile", raising=False)
    yield


# ----------------------------------------------------------------------
# is_available
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    importlib.util.find_spec("hipfile") is not None,
    reason="hipfile binding is installed on this host; can't simulate the "
    "missing-binding code path without an isolated sys.path",
)
def test_is_available_returns_false_when_binding_missing(missing_hipfile, caplog):
    caplog.set_level(logging.WARNING, logger="infera.engine.sglang.hipfile_shim")
    assert is_available() is False
    assert any("hipfile binding unavailable" in r.message for r in caplog.records)
    # The install-hint substring must be present so operators learn how
    # to fix the missing piece (LMCache-style wording).
    assert any("deploy/docker/scripts/build_hipfile.sh" in r.message for r in caplog.records)


def test_is_available_true_when_binding_present(fake_hipfile):
    assert is_available() is True
    # Exactly one native Driver should have been constructed.
    assert len(_FakeDriver.instances) == 1


# ----------------------------------------------------------------------
# HipFileDriver
# ----------------------------------------------------------------------


def test_driver_singleton(fake_hipfile):
    d1 = HipFileDriver()
    d2 = HipFileDriver()
    assert d1 is d2

    # ensure_open is idempotent — second call must NOT construct a
    # second native Driver.
    d1.ensure_open()
    d1.ensure_open()
    assert len(_FakeDriver.instances) == 1
    assert d1.is_open() is True

    # close() releases the reference; ensure_open() re-opens (new Driver).
    d1.close()
    assert d1.is_open() is False
    d1.ensure_open()
    assert len(_FakeDriver.instances) == 2


# ----------------------------------------------------------------------
# RegisteredBuffer — validation
# ----------------------------------------------------------------------


def test_registered_buffer_rejects_null_ptr():
    with pytest.raises(ValueError, match="non-NULL"):
        RegisteredBuffer(ptr=0, size=HIPFILE_ALIGNMENT)


def test_registered_buffer_rejects_zero_size():
    with pytest.raises(ValueError, match="size must be > 0"):
        RegisteredBuffer(ptr=HIPFILE_ALIGNMENT, size=0)


def test_registered_buffer_rejects_unaligned_ptr():
    with pytest.raises(ValueError, match="not 4096-byte aligned"):
        RegisteredBuffer(ptr=HIPFILE_ALIGNMENT + 1, size=HIPFILE_ALIGNMENT)


def test_registered_buffer_rejects_unaligned_size():
    with pytest.raises(ValueError, match="not a multiple of 4096"):
        RegisteredBuffer(ptr=HIPFILE_ALIGNMENT, size=HIPFILE_ALIGNMENT + 1)


# ----------------------------------------------------------------------
# RegisteredBuffer — lifecycle
# ----------------------------------------------------------------------


def test_registered_buffer_register_deregister_lifecycle(fake_hipfile):
    ptr = HIPFILE_ALIGNMENT * 100  # well-aligned
    size = HIPFILE_ALIGNMENT * 4

    with RegisteredBuffer(ptr=ptr, size=size, flags=0) as rb:
        assert rb._registered is True
        # Buffer was constructed with the exact (ptr, size, flags).
        assert _FakeBuffer.constructed == [(ptr, size, 0)]
        assert isinstance(rb._buffer, _FakeBuffer)
        assert rb._buffer.register_calls == 1
        assert rb._buffer.deregister_calls == 0
        # Stash the buffer ref so we can probe deregister after __exit__.
        buffer_ref = rb._buffer

    # __exit__ deregisters exactly once and drops the buffer ref.
    assert buffer_ref.deregister_calls == 1
    # Double-deregister via explicit call is a no-op (idempotent).
    # (The internal _buffer is None now; we just verify no second call.)
    assert buffer_ref.deregister_calls == 1


# ----------------------------------------------------------------------
# HipFile — open/close + dispatch
# ----------------------------------------------------------------------


def test_hipfile_context_manager_open_close(fake_hipfile, tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"\x00" * 4096)

    with HipFile(str(target), "r") as f:
        assert f._fd is not None
        assert isinstance(f._handle, _FakeFileHandle)
        handle = f._handle
        assert handle.closed is False

    # __exit__ closes the underlying handle AND the POSIX fd.
    assert handle.closed is True
    assert f._handle is None
    assert f._fd is None


def test_hipfile_context_manager_closes_on_exception(fake_hipfile, tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"\x00" * 4096)

    handle_captured = None
    with pytest.raises(RuntimeError, match="boom"):
        with HipFile(str(target), "r") as f:
            handle_captured = f._handle
            raise RuntimeError("boom")

    assert handle_captured is not None
    assert handle_captured.closed is True


def test_hipfile_read_write_dispatch(fake_hipfile, tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"\x00" * 4096)

    with HipFile(str(target), "r+") as f:
        handle = f._handle
        # Wire the fake's return value so we can assert the shim
        # forwards it back to the caller as an int.
        handle.next_return = 1234
        n_read = f.read(dest=0xDEADBEEF, size=4096, file_offset=8192, dest_offset=64)
        assert n_read == 1234
        assert handle.read_calls == [(0xDEADBEEF, 4096, 8192, 64)]

        handle.next_return = 4096
        n_written = f.write(src=0xCAFEBABE, size=4096, file_offset=16384, src_offset=128)
        assert n_written == 4096
        assert handle.write_calls == [(0xCAFEBABE, 4096, 16384, 128)]


def test_hipfile_invalid_mode(fake_hipfile):
    # Validation fires in __init__ — BEFORE any binding access — so
    # this must raise even with the fake binding installed, and must
    # not have touched _FakeFileHandle.
    with pytest.raises(ValueError, match="unsupported mode"):
        HipFile("/tmp/whatever", "rwx")
    assert _FakeFileHandle.constructed == []


def test_read_on_closed_handle_raises(fake_hipfile, tmp_path):
    target = tmp_path / "blob"
    target.write_bytes(b"\x00" * 4096)

    f = HipFile(str(target), "r")
    f.__enter__()
    f.__exit__(None, None, None)

    with pytest.raises(RuntimeError, match="closed handle"):
        f.read(dest=0x1000, size=4096)
    with pytest.raises(RuntimeError, match="closed handle"):
        f.write(src=0x1000, size=4096)
