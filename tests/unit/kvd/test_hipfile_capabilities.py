###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Behavioral capability checks for the ROCm hipFile Python binding.

These tests probe what the binding actually supports at runtime, so
that the v2 chunked-fusion KV-connector redesign (buffer + handle
reuse, async submit) is informed by reality rather than a wishlist.

Skips cleanly off-host (no ``import hipfile``, no GPU). When the
binding *is* importable but the driver can't open or no GPU is
visible, the tests skip with a one-line reason rather than fail —
the runtime gating already covers absent infrastructure.

This pytest module exists so the same checks can run in CI once a
hipFile-capable runner is wired up.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import os

import pytest

pytest.importorskip("torch", reason="hipfile tests need torch for device alloc")
hipfile = pytest.importorskip("hipfile", reason="hipFile python binding not installed")

import torch  # noqa: E402
from hipfile import Buffer, Driver, FileHandle  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="hipfile binding requires a real GPU device pointer (err 5013 otherwise)",
)


ALIGN = 4096
SIZE = 34 * 1024 * 1024  # Kimi connector chunk size


@pytest.fixture(scope="module")
def driver():
    d = Driver()
    try:
        d.open()
    except Exception as e:  # pragma: no cover — broken host, not a code bug
        pytest.skip(f"hipfile driver open failed: {e!r}")
    yield d
    # leave open — Driver is process-singleton-ish; teardown not needed


def _dev_buffer(size: int) -> tuple[torch.Tensor, int]:
    raw = torch.empty(size + ALIGN, dtype=torch.uint8, device="cuda")
    pad = (-raw.data_ptr()) % ALIGN
    v = raw[pad : pad + size]
    return v, v.data_ptr()


def test_b1_reuse_buffer_across_files(driver, tmp_path):
    """One registered Buffer should write to many distinct files."""
    _, ptr = _dev_buffer(SIZE)
    buf = Buffer(ptr, SIZE, 0)
    buf.register()
    try:
        for i in range(3):
            p = tmp_path / f"f{i}.bin"
            fh = FileHandle(str(p), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            fh.open()
            n = fh.write(buf, SIZE, 0, 0)
            fh.close()
            assert n == SIZE
            assert p.stat().st_size == SIZE
    finally:
        buf.deregister()


def test_b2_single_handle_multi_offset(driver, tmp_path):
    """One FileHandle should accept multiple writes at different offsets."""
    _, ptr = _dev_buffer(SIZE)
    buf = Buffer(ptr, SIZE, 0)
    buf.register()
    p = tmp_path / "multi.bin"
    p.write_bytes(b"\x00" * (SIZE * 3))  # pre-allocate
    fh = FileHandle(str(p), os.O_WRONLY, 0o644)
    fh.open()
    try:
        for i in range(3):
            n = fh.write(buf, SIZE, i * SIZE, 0)
            assert n == SIZE
    finally:
        fh.close()
        buf.deregister()
    assert p.stat().st_size == SIZE * 3


def test_b3_threadsafe_parallel_diff_buffers_diff_files(driver, tmp_path):
    """4 threads, each with its own Buffer+FileHandle, write in parallel."""
    bufs: list[Buffer] = []
    tensors: list[torch.Tensor] = []
    try:
        for _ in range(4):
            t, p = _dev_buffer(SIZE)
            tensors.append(t)
            b = Buffer(p, SIZE, 0)
            b.register()
            bufs.append(b)
        paths = [tmp_path / f"t{i}.bin" for i in range(4)]

        def _one(i: int) -> int:
            fh = FileHandle(str(paths[i]), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            fh.open()
            n = fh.write(bufs[i], SIZE, 0, 0)
            fh.close()
            return n

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            ns = list(ex.map(_one, range(4)))
        assert all(n == SIZE for n in ns)
        assert all(p.stat().st_size == SIZE for p in paths)
    finally:
        for b in bufs:
            b.deregister()


def test_b4_write_does_not_accept_cuda_stream():
    """Documents that hipfile 0.2.0 has no stream-overlap parameter.

    The C layer has hipFileAsyncNotSupported error code in its enums
    but the Python binding's FileHandle.write signature has no stream
    parameter. We assert the negative so a future binding upgrade that
    DOES expose async-write trips this test and forces a redesign.
    """
    sig = inspect.signature(FileHandle.write)
    params = list(sig.parameters.keys())
    assert not any("stream" in n.lower() for n in params), (
        f"FileHandle.write now exposes a stream-like param: {params!r} — "
        "the chunked-fusion connector can finally use cuFileWriteAsync; "
        "update kvd_adapter/the v2 connector accordingly."
    )


def test_b5_shared_handle_two_threads(driver, tmp_path):
    """Two threads writing to the same FileHandle (different offsets) at once."""
    _, ptr = _dev_buffer(SIZE)
    buf = Buffer(ptr, SIZE, 0)
    buf.register()
    p = tmp_path / "shared.bin"
    p.write_bytes(b"\x00" * (SIZE * 2))
    fh = FileHandle(str(p), os.O_WRONLY, 0o644)
    fh.open()
    try:

        def _one(i: int) -> int:
            return fh.write(buf, SIZE, i * SIZE, 0)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
            ns = list(ex.map(_one, range(2)))
        assert all(n == SIZE for n in ns)
    finally:
        fh.close()
        buf.deregister()
    assert p.stat().st_size == SIZE * 2
