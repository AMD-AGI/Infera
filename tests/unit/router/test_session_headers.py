###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Agent-session headers: the server stashes them on the body, the PD router
sends them on both legs (concurrent and serial-pull), and no router forwards
the stash itself in a worker body."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from infera.common.nats_request import TYPE_DATA, TYPE_DONE
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.base import BaseRouter
from infera.router.disagg import DisaggRouter
from infera.router.mixed import MixedRouter
from infera.router.policy.round_robin import RoundRobinPolicy
from infera.router.policy.session_affinity import SessionAffinityPolicy
from infera.router.policy.target import RouteTarget
from infera.server import app as app_module
from infera.server.app import init_app

SESSION = {
    "x-correlation-id": "corr-1",
    "x-session-id": "sess-1",
}


def _worker(
    wid,
    mode=DisaggMode.MIXED,
    protocol=None,
    params=None,
    engine=EngineType.SGLANG,
    transport="http",
):
    meta = {"protocol": protocol, "params": params or {}} if protocol else {}
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=engine,
        disagg_mode=mode,
        disagg_meta=meta,
        request_transport=transport,
    )


class _Pool:
    def __init__(self, workers):
        self._workers = workers

    def list_active(self, model=None, mode=None):
        return [w for w in self._workers if mode is None or w.disagg_mode == mode]


class _FirstPolicy:
    def pick(self, candidates, body, role_hint=None):
        return RouteTarget(candidates[0]), []

    def on_request_started(self, route_key, blocks):
        pass

    def on_request_finished(self, route_key, blocks):
        pass

    def on_request_failed(self, route_key, request, role_hint=None):
        pass


def _capture(router, reply=None):
    """Record each worker request as captured[host] = (headers, body)."""
    captured: dict[str, tuple[dict, dict]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured[request.url.host] = (dict(request.headers), json.loads(request.content))
        return httpx.Response(200, json=reply or {"id": request.url.host})

    router._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return captured


def _sglang_pair():
    p = _worker("p1", DisaggMode.PREFILL, "sglang-bootstrap", {"bootstrap_addr": "boot:8998"})
    d = _worker("d1", DisaggMode.DECODE, "sglang-bootstrap")
    return [p, d]


def _atom_pair():
    params = {"host": "h", "http_port": 1, "handshake_port": 2, "tp_size": 1, "dp_size": 1}
    p = _worker("p1", DisaggMode.PREFILL, "atom-mooncake", params, EngineType.ATOM)
    d = _worker("d1", DisaggMode.DECODE, "atom-mooncake", params, EngineType.ATOM)
    return [p, d]


# Concurrent (SGLang bootstrap) and serial-pull (ATOM Mooncake) dispatchers.
_TOPOLOGIES = [
    pytest.param(_sglang_pair, None, id="concurrent"),
    pytest.param(_atom_pair, {"kv_transfer_params": {"transfer_id": "t"}}, id="serial-pull"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("pair", "reply"), _TOPOLOGIES)
async def test_pd_forwards_session_headers_to_both_legs(pair, reply):
    r = DisaggRouter(_Pool(pair()), _FirstPolicy())
    captured = _capture(r, reply)
    body = {"model": "m", "prompt": "hi", "_infera_session_headers": dict(SESSION)}

    resp = await r.dispatch(body, stream=False, path="/v1/completions")

    assert resp.status_code == 200
    assert set(captured) == {"p1", "d1"}
    for headers, leg_body in captured.values():
        assert {k: headers[k] for k in SESSION} == SESSION
        assert "_infera_session_headers" not in leg_body
    await r.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("decode_status", [200, 500])
async def test_pd_failure_drops_the_session_bindings(decode_status):
    policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="both")
    r = DisaggRouter(_Pool(_sglang_pair()), policy)

    def handler(request: httpx.Request) -> httpx.Response:
        status = decode_status if request.url.host == "d1" else 200
        return httpx.Response(status, json={"id": request.url.host})

    r._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    body = {"model": "m", "prompt": "hi", "_infera_session_headers": dict(SESSION)}

    resp = await r.dispatch(body, stream=False, path="/v1/completions")

    assert resp.status_code == decode_status
    roles = {key[2] for key in policy._bindings}
    assert roles == (set() if decode_status >= 400 else {"prefill", "decode"})
    await r.aclose()


class _RecordingNats:
    """Fake NATS request client: records each leg's payload, replies 200."""

    def __init__(self) -> None:
        self.payloads: dict[str, dict] = {}

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        self.payloads[worker_id] = payload
        yield (TYPE_DATA, None, json.dumps({"id": worker_id}).encode())
        yield (TYPE_DONE, 200, b"")


@pytest.mark.asyncio
async def test_pd_nats_forwards_session_headers_to_both_legs():
    boot = {"bootstrap_addr": "boot:8998"}
    p = _worker("p1", DisaggMode.PREFILL, "sglang-bootstrap", boot, transport="nats")
    d = _worker("d1", DisaggMode.DECODE, "sglang-bootstrap", transport="nats")
    nats = _RecordingNats()
    r = DisaggRouter(_Pool([p, d]), _FirstPolicy(), nats_client=nats)
    body = {"model": "m", "prompt": "hi", "_infera_session_headers": dict(SESSION)}

    resp = await r.dispatch(body, stream=False, path="/v1/completions")
    for _ in range(10):  # let the detached prefill drain run
        await asyncio.sleep(0)

    assert resp.status_code == 200
    assert set(nats.payloads) == {"p1", "d1"}
    for payload in nats.payloads.values():
        assert {k: payload["headers"][k] for k in SESSION} == SESSION
        assert "_infera_session_headers" not in payload["body"]
    await r.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("pair", "reply"), _TOPOLOGIES)
async def test_pd_without_session_headers_sends_none(pair, reply):
    r = DisaggRouter(_Pool(pair()), _FirstPolicy())
    captured = _capture(r, reply)

    resp = await r.dispatch({"model": "m", "prompt": "hi"}, stream=False, path="/v1/completions")

    assert resp.status_code == 200
    for headers, _ in captured.values():
        assert not set(SESSION) & set(headers)
    await r.aclose()


@pytest.mark.asyncio
async def test_mixed_strips_session_headers_from_body():
    r = MixedRouter(_Pool([_worker("w1")]), _FirstPolicy())
    captured = _capture(r)

    resp = await r.dispatch({"model": "m", "_infera_session_headers": dict(SESSION)}, stream=False)

    assert resp.status_code == 200
    assert "_infera_session_headers" not in captured["w1"][1]
    await r.aclose()


# --- server front door --------------------------------------------------------


class _RecordingRouter:
    def __init__(self) -> None:
        self.body: dict | None = None

    async def aclose(self) -> None:
        return

    async def dispatch(self, body, *, stream, path):
        self.body = body
        return JSONResponse(content={"choices": [], "usage": {}})


BaseRouter.register(_RecordingRouter)


@pytest.fixture
def client_and_router():
    router = _RecordingRouter()
    init_app(reg=None, rtr=router, kv=None)  # type: ignore[arg-type]
    yield TestClient(app_module.app), router


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/completions", {"model": "m", "prompt": "hi"}),
        ("/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]}),
        (
            "/v1/messages",
            {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]},
        ),
    ],
)
def test_server_stashes_session_headers(client_and_router, path, payload):
    client, router = client_and_router
    headers = {"X-Correlation-ID": "corr-1", "X-Session-ID": "sess-1", "X-Other": "x"}

    assert client.post(path, json=payload, headers=headers).status_code == 200

    assert router.body["_infera_session_headers"] == {
        "x-correlation-id": "corr-1",
        "x-session-id": "sess-1",
    }


def test_server_omits_stash_without_session_headers(client_and_router):
    client, router = client_and_router

    client.post("/v1/completions", json={"model": "m", "prompt": "hi"})

    assert "_infera_session_headers" not in router.body


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        ("/v1/completions", {"model": "m", "prompt": "hi"}),
        (
            "/v1/messages",
            {"model": "m", "max_tokens": 4, "messages": [{"role": "user", "content": "hi"}]},
        ),
    ],
)
def test_server_rejects_bad_session_ids_only_with_affinity_on(client_and_router, path, payload):
    client, router = client_and_router
    sid = "X-Session-ID"
    bad = [
        [(sid, "a"), (sid, "b")],
        [(sid, "")],
        [(sid, "x" * 1025)],
        [(sid, "caf\xe9".encode("latin-1"))],
    ]
    for headers in bad:
        assert client.post(path, json=payload, headers=headers).status_code == 200

    router.policy = SessionAffinityPolicy(RoundRobinPolicy(), mode="prefill")
    for headers in bad:
        assert client.post(path, json=payload, headers=headers).status_code == 400
    assert client.post(path, json=payload, headers={sid: "x" * 1024}).status_code == 200


def test_server_drops_client_stash_and_unsendable_values(client_and_router):
    """A client cannot inject leg headers through the body, and a value httpx
    cannot encode never reaches the PD legs."""
    client, router = client_and_router
    payload = {"model": "m", "prompt": "hi", "_infera_session_headers": {"x-evil": "1"}}

    client.post("/v1/completions", json=payload)
    assert "_infera_session_headers" not in router.body

    headers = {"X-Session-ID": "caf\xe9".encode("latin-1"), "X-Correlation-ID": "corr-1"}
    client.post("/v1/completions", json=payload, headers=headers)
    assert router.body["_infera_session_headers"] == {"x-correlation-id": "corr-1"}
