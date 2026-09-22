###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""CLI plumbing for --wait-for-decode on the SGLang worker."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("sglang")

from infera.engine.base import EngineDeath  # noqa: E402
from infera.engine.sglang.__main__ import (  # noqa: E402
    _run_started_engine,
    _wait_for_decode_until_stop,
)
from infera.engine.sglang.args import parse_sglang_args  # noqa: E402

_PREFILL = [
    "--model-path",
    "Qwen/Qwen3-0.6B",
    "--served-model-name",
    "glm-5-3",
    "--disaggregation-mode",
    "prefill",
]


def test_wait_for_decode_defaults_none_on_prefill():
    args = parse_sglang_args(_PREFILL)
    assert args.wait_for_decode is None
    assert args.decode_ready_timeout is None
    assert args.k8s_label_selector is None


def test_no_wait_for_decode_flag():
    args = parse_sglang_args([*_PREFILL, "--no-wait-for-decode"])
    assert args.wait_for_decode is False


def test_wait_for_decode_timeout_and_selector():
    args = parse_sglang_args(
        [
            *_PREFILL,
            "--wait-for-decode",
            "--decode-ready-timeout",
            "90",
            "--k8s-label-selector",
            "infera.amd.com/deployment=x",
        ]
    )
    assert args.wait_for_decode is True
    assert args.decode_ready_timeout == 90.0
    assert args.k8s_label_selector == "infera.amd.com/deployment=x"


@pytest.mark.asyncio
async def test_wait_for_decode_until_stop_aborts_when_engine_dies(monkeypatch):
    async def _hang(*_a, **_k):
        await asyncio.Event().wait()

    monkeypatch.setattr("infera.engine.sglang.__main__._maybe_wait_for_decode", _hang)
    stop = asyncio.Event()

    async def _trip():
        await asyncio.sleep(0.01)
        stop.set()

    asyncio.create_task(_trip())
    assert await _wait_for_decode_until_stop(SimpleNamespace(), SimpleNamespace(), stop) is False


@pytest.mark.asyncio
async def test_wait_for_decode_until_stop_returns_true_on_success(monkeypatch):
    async def _ok(*_a, **_k):
        return None

    monkeypatch.setattr("infera.engine.sglang.__main__._maybe_wait_for_decode", _ok)
    assert (
        await _wait_for_decode_until_stop(SimpleNamespace(), SimpleNamespace(), asyncio.Event())
        is True
    )


@pytest.mark.asyncio
async def test_started_engine_keeps_cancelled_error_and_forces_cleanup(monkeypatch):
    stopped = []
    killed = []

    class _Engine:
        async def stop(self):
            stopped.append(True)

    async def _cancelled(*_args):
        raise asyncio.CancelledError

    async def _watch():
        await asyncio.Event().wait()

    death_task = asyncio.create_task(_watch())
    monkeypatch.setattr(
        "infera.engine.sglang.__main__._supervise_engine",
        lambda _engine: (asyncio.Event(), EngineDeath(exit_status=9), death_task),
    )
    monkeypatch.setattr(
        "infera.engine.sglang.__main__._wait_for_decode_until_stop",
        _cancelled,
    )
    monkeypatch.setattr(
        "infera.engine.sglang.__main__._kill_process_group_safely",
        lambda: killed.append(True),
    )

    with pytest.raises(asyncio.CancelledError):
        await _run_started_engine(SimpleNamespace(), _Engine(), SimpleNamespace())

    assert stopped == [True]
    assert killed == [True]
