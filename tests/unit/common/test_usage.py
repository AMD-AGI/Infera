# SPDX-License-Identifier: MIT
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Privacy, lifecycle and transport guarantees; never contact a real collector."""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from infera.common import usage


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    for key in (
        "INFERA_NO_USAGE_STATS",
        "DO_NOT_TRACK",
        "TELEMETRY_DISABLED",
        "INFERA_USAGE_STATS_SERVER",
        *usage._CI_VARS,
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setattr(usage, "DEFAULT_ENDPOINT", "")
    # pytest restores its phase marker after fixture setup. Exercise that
    # marker explicitly in test_ci_disabled, not in transport tests.
    monkeypatch.setattr(
        usage, "_CI_VARS", tuple(k for k in usage._CI_VARS if k != "PYTEST_CURRENT_TEST")
    )


@pytest.mark.parametrize("key", ["INFERA_NO_USAGE_STATS", "DO_NOT_TRACK", "TELEMETRY_DISABLED"])
@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_opt_out_wins(monkeypatch, key, value):
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example/events")
    monkeypatch.setenv(key, value)
    assert not usage.usage_enabled()


def test_default_enabled_and_file_opt_out(monkeypatch, tmp_path):
    assert usage.usage_enabled()
    (tmp_path / "infera").mkdir()
    (tmp_path / "infera" / "do_not_track").touch()
    assert not usage.usage_enabled()
    (tmp_path / "infera" / "do_not_track").unlink()
    assert usage.usage_enabled()


def test_home_fallback(monkeypatch, tmp_path):
    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setenv("HOME", str(tmp_path))
    path = tmp_path / ".config" / "infera"
    path.mkdir(parents=True)
    (path / "do_not_track").touch()
    assert not usage.usage_enabled()


@pytest.mark.parametrize("key", usage._CI_VARS)
def test_ci_disabled(monkeypatch, key):
    monkeypatch.setattr(usage, "_CI_VARS", (key,))
    monkeypatch.setenv(key, "running")
    assert not usage.usage_enabled()


@pytest.mark.parametrize(
    "endpoint",
    [
        "",
        "http://collector.example",
        "https://",
        "https://[bad",
        "https://host:bad",
        "https://user:secret@host",
        "https://host/#fragment",
    ],
)
def test_invalid_endpoint(monkeypatch, endpoint):
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", endpoint)
    assert usage.usage_endpoint() == ""


def test_https_endpoint(monkeypatch):
    endpoint = "https://collector.example/events"
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", endpoint)
    assert usage.usage_endpoint() == endpoint


def test_metadata_allowlist(monkeypatch):
    monkeypatch.setattr(usage, "version", lambda _: "1.2.3+secret-branch")
    monkeypatch.setattr(usage.platform, "machine", lambda: "private-hardware-name")
    data = usage._metadata("worker", "custom/private-model", "/secret/path", "https://secret")
    assert data == {
        "infera_version": "1.2.3",
        "python_version": usage.platform.python_version(),
        "os": usage.platform.system(),
        "cpu_architecture": "unknown",
        "component": "worker",
        "engine": "unknown",
        "disagg_mode": "unknown",
        "request_transport": "unknown",
    }
    assert "secret" not in str(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [True, False])
async def test_no_collection_without_endpoint_or_when_disabled(monkeypatch, disabled):
    if disabled:
        monkeypatch.setenv("INFERA_NO_USAGE_STATS", "1")
        monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example")
    metadata = AsyncMock(side_effect=AssertionError("must not collect"))
    monkeypatch.setattr(usage, "_metadata", metadata)
    async with usage.usage_session("server"):
        pass
    metadata.assert_not_called()


@pytest.mark.asyncio
async def test_reporting_contract_and_shutdown(monkeypatch):
    import json

    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example/events")
    monkeypatch.setattr(usage, "HEARTBEAT_SECONDS", 0.001)
    reports = []
    received = asyncio.Event()

    def handler(request):
        assert request.method == "POST"
        reports.append(json.loads(request.content))
        if len(reports) >= 2:
            received.set()
        # Redirects must never forward deployment metadata elsewhere.
        return httpx.Response(302, headers={"Location": "https://other.example"})

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        usage.httpx,
        "AsyncClient",
        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw),
    )
    async with usage.usage_session("worker", engine="vllm", mode="decode", transport="nats"):
        await asyncio.wait_for(received.wait(), 1)
    assert reports[0]["event"] == "session_start"
    assert reports[1]["event"] == "heartbeat"
    assert reports[0]["session_id"] == reports[1]["session_id"]
    assert reports[1]["sequence"] == 1
    assert reports[0]["engine"] == "vllm"
    assert reports[0]["disagg_mode"] == "decode"
    assert reports[0]["request_transport"] == "nats"
    assert set(reports[0]) == {
        "schema_version",
        "event",
        "session_id",
        "sequence",
        "elapsed_seconds",
        "infera_version",
        "python_version",
        "os",
        "cpu_architecture",
        "component",
        "engine",
        "disagg_mode",
        "request_transport",
    }
    assert not any(t.get_name() == "infera-usage" for t in asyncio.all_tasks())


@pytest.mark.asyncio
async def test_dynamic_opt_out_stops_heartbeats(monkeypatch):
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example")
    monkeypatch.setattr(usage, "HEARTBEAT_SECONDS", 0)
    reports = []

    def handler(request):
        reports.append(request)
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        return httpx.Response(200)

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        usage.httpx,
        "AsyncClient",
        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw),
    )
    await asyncio.wait_for(usage._report(usage.usage_endpoint(), {}), 1)
    assert len(reports) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["network", "timeout", "construction"])
async def test_transport_failure_does_not_break_serving(monkeypatch, failure):
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example")
    monkeypatch.setattr(usage, "SEND_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(usage, "HEARTBEAT_SECONDS", 0)
    attempted = asyncio.Event()

    async def handler(request):
        attempted.set()
        monkeypatch.setenv("DO_NOT_TRACK", "1")
        if failure == "timeout":
            await asyncio.Event().wait()
        raise httpx.ConnectError("private URL and proxy credentials")

    client_type = httpx.AsyncClient

    def factory(**kwargs):
        if failure == "construction":
            attempted.set()
            raise RuntimeError("bad proxy configuration")
        return client_type(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(usage.httpx, "AsyncClient", factory)
    async with usage.usage_session("server"):
        await asyncio.wait_for(attempted.wait(), 1)
        task = next((t for t in asyncio.all_tasks() if t.get_name() == "infera-usage"), None)
        if task:
            await asyncio.wait_for(asyncio.shield(task), 1)


@pytest.mark.asyncio
async def test_serving_exception_propagates_and_cancels_reporter(monkeypatch):
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example")
    reporter = AsyncMock(side_effect=lambda *args: None)
    monkeypatch.setattr(usage, "_report", reporter)
    with pytest.raises(ValueError, match="serving failed"):
        async with usage.usage_session("server"):
            raise ValueError("serving failed")
    assert not any(t.get_name() == "infera-usage" for t in asyncio.all_tasks())


@pytest.mark.asyncio
async def test_shutdown_cancels_inflight_send(monkeypatch):
    monkeypatch.setenv("INFERA_USAGE_STATS_SERVER", "https://collector.example")
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    client_type = httpx.AsyncClient
    monkeypatch.setattr(
        usage.httpx,
        "AsyncClient",
        lambda **kw: client_type(transport=httpx.MockTransport(handler), **kw),
    )
    async with usage.usage_session("server"):
        await asyncio.wait_for(entered.wait(), 1)
    assert cancelled.is_set()
    assert not any(t.get_name() == "infera-usage" for t in asyncio.all_tasks())
