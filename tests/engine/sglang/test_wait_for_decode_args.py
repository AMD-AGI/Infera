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
from infera.engine.decode_barrier import DEFAULT_PD_PROBE_TIMEOUT  # noqa: E402
from infera.engine.sglang.__main__ import (  # noqa: E402
    _maybe_wait_for_decode,
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
async def test_maybe_wait_for_decode_bounds_the_probe_with_the_shared_deadline(monkeypatch):
    """The KV probe spends the decode-ready budget, it does not extend it."""

    async def _found(*_a, **_k):
        return {"url": "http://decode:30000", "dp_size": 1}

    async def _hang(*_a, **_k):
        await asyncio.Event().wait()

    monkeypatch.delenv("LWS_WORKER_INDEX", raising=False)
    monkeypatch.setattr("infera.engine.sglang.__main__.wait_for_decode", _found)
    monkeypatch.setattr("infera.engine.sglang.__main__.verify_pd_peer", _hang)

    args = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_mode="prefill",
            node_rank=0,
            served_model_name="glm-5-3",
            dp_size=1,
            disaggregation_bootstrap_port=8998,
        ),
        wait_for_decode=True,
        decode_ready_timeout=0.05,
        discovery_backend="etcd",
        etcd_endpoint="host:2379",
        etcd_prefix="/infera/",
        k8s_namespace=None,
        k8s_label_selector=None,
    )
    config = SimpleNamespace(host="10.0.0.1", port=30000)

    started = asyncio.get_running_loop().time()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_maybe_wait_for_decode(args, config), timeout=5.0)
    assert asyncio.get_running_loop().time() - started < 1.0


def _prefill_args(**overrides):
    args = SimpleNamespace(
        server_args=SimpleNamespace(
            disaggregation_mode="prefill",
            node_rank=0,
            served_model_name="glm-5-3",
            dp_size=1,
            disaggregation_bootstrap_port=8998,
        ),
        wait_for_decode=True,
        decode_ready_timeout=None,
        discovery_backend="kubernetes",
        etcd_endpoint=None,
        etcd_prefix="/infera/",
        k8s_namespace="ns0",
        k8s_label_selector=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


async def _record_barrier_budgets(monkeypatch, total_timeout: float) -> dict:
    """Run the barrier with stubbed steps and capture the budget each one got."""
    seen: dict[str, float] = {}

    class _Client:
        async def aclose(self):
            return None

    async def _selector(*_a, timeout, **_k):
        seen["selector"] = timeout
        return "infera.amd.com/deployment=idep-a"

    async def _decode(*_a, timeout, **_k):
        seen["decode"] = timeout
        return {"url": "http://decode:30000", "dp_size": 1}

    async def _probe(*_a, **_k):
        return None

    real_wait_for = asyncio.wait_for

    async def _wait_for(awaitable, timeout):
        seen["probe"] = timeout
        return await real_wait_for(awaitable, timeout)

    monkeypatch.delenv("LWS_WORKER_INDEX", raising=False)
    monkeypatch.setattr("infera.engine.sglang.__main__.make_client", lambda **_k: _Client())
    monkeypatch.setattr("infera.engine.sglang.__main__.wait_for_k8s_label_selector", _selector)
    monkeypatch.setattr("infera.engine.sglang.__main__.wait_for_decode", _decode)
    monkeypatch.setattr("infera.engine.sglang.__main__.verify_pd_peer", _probe)
    monkeypatch.setattr(asyncio, "wait_for", _wait_for)

    args = _prefill_args(decode_ready_timeout=total_timeout)
    await _maybe_wait_for_decode(args, SimpleNamespace(host="10.0.0.1", port=30000))
    return seen


@pytest.mark.asyncio
async def test_maybe_wait_for_decode_reserves_a_probe_slice_of_the_budget(monkeypatch):
    """Selector and decode lookup stop early so the probe keeps its reserve."""
    total = DEFAULT_PD_PROBE_TIMEOUT * 4
    seen = await _record_barrier_budgets(monkeypatch, total)

    discovery = total - DEFAULT_PD_PROBE_TIMEOUT
    assert seen["selector"] == pytest.approx(discovery, abs=1.0)
    assert seen["decode"] == pytest.approx(discovery, abs=1.0)
    assert seen["probe"] == pytest.approx(total, abs=1.0)


@pytest.mark.asyncio
async def test_maybe_wait_for_decode_gives_a_short_budget_to_the_probe(monkeypatch):
    """Below the reserve, discovery relies on its single immediate lookup."""
    total = DEFAULT_PD_PROBE_TIMEOUT / 2
    seen = await _record_barrier_budgets(monkeypatch, total)

    assert seen["selector"] == 0.0
    assert seen["decode"] == 0.0
    assert seen["probe"] == pytest.approx(total, abs=1.0)


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
