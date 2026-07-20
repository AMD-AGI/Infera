###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for the unified profiling control plane.

Covers the frontend routes POST /v1/admin/profile/{start,stop} (target
selection, disabled->403, no-match->404, bad-role->400) and the profiling
module's fan-out helper (per-worker URL + aggregate result), using
httpx.ASGITransport / httpx.MockTransport so no real network is involved.
"""

from __future__ import annotations

import httpx
import pytest

from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo, WorkerStatus
from infera.server import app as app_module
from infera.server.app import app, init_app
from infera.server.profiling import fan_out_profile, select_targets


class _FakeRouter:
    async def aclose(self):
        pass


class _FakeRegistry:
    """Minimal Registry stand-in: only list_all() is used by the routes."""

    def __init__(self, workers: list[WorkerInfo]):
        self._workers = workers

    def list_all(self) -> list[WorkerInfo]:
        return list(self._workers)


def _workers() -> list[WorkerInfo]:
    return [
        WorkerInfo(
            worker_id="w-sglang",
            url="http://10.0.0.1:30000",
            model_name="Qwen/Qwen3-0.6B",
            engine=EngineType.SGLANG,
            disagg_mode=DisaggMode.MIXED,
        ),
        WorkerInfo(
            worker_id="w-prefill",
            url="http://10.0.0.2:30000",
            model_name="Qwen/Qwen3-0.6B",
            engine=EngineType.SGLANG,
            disagg_mode=DisaggMode.PREFILL,
        ),
        WorkerInfo(
            worker_id="w-vllm",
            url="http://10.0.0.3:30000",
            model_name="other/model",
            engine=EngineType.VLLM,
            disagg_mode=DisaggMode.DECODE,
        ),
        WorkerInfo(
            worker_id="w-dead",
            url="http://10.0.0.4:30000",
            model_name="Qwen/Qwen3-0.6B",
            engine=EngineType.SGLANG,
            status=WorkerStatus.DEAD,
        ),
    ]


def _init(enable_profiling: bool, workers: list[WorkerInfo] | None = None) -> None:
    init_app(
        reg=_FakeRegistry(workers if workers is not None else _workers()),  # type: ignore[arg-type]
        rtr=_FakeRouter(),  # type: ignore[arg-type]
        kv=None,
        kvd_socket_path=None,
        enable_profiling=enable_profiling,
    )


async def _post(path: str, json_body: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        return await ac.post(path, json=json_body if json_body is not None else {})


# --- route: disabled by default -------------------------------------------


@pytest.mark.asyncio
async def test_profile_disabled_returns_403():
    _init(enable_profiling=False)
    r = await _post("/v1/admin/profile/start")
    assert r.status_code == 403
    assert "enable-profiling" in r.text


# --- route: broadcast + filtering -----------------------------------------


@pytest.mark.asyncio
async def test_profile_broadcast_hits_all_active(monkeypatch):
    _init(enable_profiling=True)
    seen: dict = {}

    async def _fake_fan_out(client, workers, action, body=None):
        seen["ids"] = [w.worker_id for w in workers]
        seen["action"] = action
        seen["body"] = body
        return {"action": action, "requested": len(workers), "succeeded": len(workers)}

    monkeypatch.setattr(app_module, "fan_out_profile", _fake_fan_out)
    r = await _post("/v1/admin/profile/start", {"num_steps": 3})
    assert r.status_code == 200, r.text
    # Dead worker excluded; the three ACTIVE workers all targeted.
    assert set(seen["ids"]) == {"w-sglang", "w-prefill", "w-vllm"}
    assert seen["action"] == "start"
    # Engine passthrough: non-selector body keys forwarded verbatim.
    assert seen["body"] == {"num_steps": 3}


@pytest.mark.asyncio
async def test_profile_filter_by_worker_id(monkeypatch):
    _init(enable_profiling=True)
    seen: dict = {}

    async def _fake_fan_out(client, workers, action, body=None):
        seen["ids"] = [w.worker_id for w in workers]
        return {"requested": len(workers)}

    monkeypatch.setattr(app_module, "fan_out_profile", _fake_fan_out)
    r = await _post("/v1/admin/profile/start", {"worker_id": "w-prefill"})
    assert r.status_code == 200, r.text
    assert seen["ids"] == ["w-prefill"]


@pytest.mark.asyncio
async def test_profile_filter_by_role_and_model(monkeypatch):
    _init(enable_profiling=True)
    seen: dict = {}

    async def _fake_fan_out(client, workers, action, body=None):
        seen["ids"] = [w.worker_id for w in workers]
        return {"requested": len(workers)}

    monkeypatch.setattr(app_module, "fan_out_profile", _fake_fan_out)
    # role=prefill matches only w-prefill.
    r = await _post("/v1/admin/profile/start", {"role": "prefill"})
    assert r.status_code == 200, r.text
    assert seen["ids"] == ["w-prefill"]


@pytest.mark.asyncio
async def test_profile_no_match_returns_404(monkeypatch):
    _init(enable_profiling=True)
    monkeypatch.setattr(
        app_module, "fan_out_profile", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    r = await _post("/v1/admin/profile/start", {"worker_id": "does-not-exist"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_profile_bad_role_returns_400():
    _init(enable_profiling=True)
    r = await _post("/v1/admin/profile/start", {"role": "bogus"})
    assert r.status_code == 400
    assert "role" in r.text


# --- module: select_targets -----------------------------------------------


def test_select_targets_excludes_dead_and_defaults_to_broadcast():
    ids = [w.worker_id for w in select_targets(_workers())]
    assert set(ids) == {"w-sglang", "w-prefill", "w-vllm"}


def test_select_targets_role_filter():
    ids = [w.worker_id for w in select_targets(_workers(), role="decode")]
    assert ids == ["w-vllm"]


def test_select_targets_invalid_role_raises():
    with pytest.raises(ValueError):
        select_targets(_workers(), role="nope")


# --- module: fan_out_profile ----------------------------------------------


@pytest.mark.asyncio
async def test_fan_out_posts_to_engine_endpoint_and_aggregates():
    posted: list[str] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        posted.append(str(request.url))
        # Fail the vllm worker to exercise partial-failure aggregation.
        if "10.0.0.3" in str(request.url):
            return httpx.Response(400, json={"error": "no VLLM_TORCH_PROFILER_DIR"})
        return httpx.Response(200, json={"ok": True})

    workers = select_targets(_workers())  # the three active workers
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        agg = await fan_out_profile(client, workers, "start", {"num_steps": 2})

    # Every active worker hit at its engine /start_profile.
    assert sorted(posted) == [
        "http://10.0.0.1:30000/start_profile",
        "http://10.0.0.2:30000/start_profile",
        "http://10.0.0.3:30000/start_profile",
    ]
    assert agg["action"] == "start"
    assert agg["requested"] == 3
    assert agg["succeeded"] == 2  # vllm one returned 400


@pytest.mark.asyncio
async def test_fan_out_survives_transport_error():
    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    workers = select_targets(_workers(), worker_id="w-sglang")
    transport = httpx.MockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        agg = await fan_out_profile(client, workers, "stop")

    assert agg["requested"] == 1
    assert agg["succeeded"] == 0
    assert agg["results"][0]["ok"] is False
