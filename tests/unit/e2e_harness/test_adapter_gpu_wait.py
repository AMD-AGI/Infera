###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Regression tests for the mixed-worker GPU-free barrier."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tests.e2e.harness import adapter
from tests.e2e.harness.gpu_cleanup import GPU_DIRTY_MARKER


@pytest.fixture
def emitted(monkeypatch):
    lines = []
    monkeypatch.setattr(adapter, "_emit", lines.append)
    return lines


def test_gpu_vram_treats_valid_non_object_json_as_unreadable(monkeypatch):
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="null"),
    )

    assert adapter._gpu_vram() == {}


def test_gpu_vram_treats_nonzero_rocm_smi_as_unreadable(monkeypatch):
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout='{"card0": {"VRAM Total Memory (B)": "100", '
            '"VRAM Total Used Memory (B)": "99"}}',
        ),
    )

    assert adapter._gpu_vram() == {}


@pytest.mark.asyncio
async def test_gpu_wait_is_a_noop_when_rocm_smi_is_unreadable(monkeypatch):
    monkeypatch.setattr(adapter, "_gpu_vram", lambda: {})

    await adapter._await_gpus_freed([0], timeout=0)


@pytest.mark.asyncio
async def test_gpu_wait_returns_when_vram_is_below_five_percent(monkeypatch):
    monkeypatch.setattr(adapter, "_gpu_vram", lambda: {0: (1, 100)})

    await adapter._await_gpus_freed([0], timeout=0)


@pytest.mark.asyncio
async def test_gpu_wait_observes_busy_then_free(monkeypatch):
    readings = iter(({0: (99, 100)}, {0: (1, 100)}))

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(adapter, "_gpu_vram", lambda: next(readings))
    monkeypatch.setattr(adapter.asyncio, "sleep", no_sleep)

    await adapter._await_gpus_freed([0], timeout=10)


@pytest.mark.asyncio
async def test_gpu_wait_fails_instead_of_launching_on_persistently_busy_vram(monkeypatch, emitted):
    monkeypatch.setenv("INFERA_E2E_SLURM_NODE", "node-a")
    monkeypatch.setattr(adapter, "_gpu_vram", lambda: {0: (270 * 1024**3, 288 * 1024**3)})

    with pytest.raises(RuntimeError, match=rf"{GPU_DIRTY_MARKER}.*gpu0=270.0/288.0GiB"):
        await adapter._await_gpus_freed([0], timeout=0)
    assert emitted == [
        "INFERA_E2E_GPU_NODE_DIRTY_NODE=node-a "
        "GPU(s) remained busy or unreported after 0s (gpu0=270.0/288.0GiB)"
    ]


@pytest.mark.asyncio
async def test_gpu_wait_fails_when_readable_result_omits_requested_gpu(monkeypatch, emitted):
    monkeypatch.setattr(adapter, "_gpu_vram", lambda: {0: (1, 100)})

    with pytest.raises(RuntimeError, match=rf"{GPU_DIRTY_MARKER}.*missing=\[1\]"):
        await adapter._await_gpus_freed([1], timeout=0)
