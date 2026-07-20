###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

from infera.common.discovery import Registry
from infera.common.worker_pool import DisaggMode, WorkerStatus
from infera.kvd.client import KvdClient, KvdConnectionError
from infera.router.base import BaseRouter
from infera.router.kv_event.client import KvEventClient
from infera.router.policy.target import expand_targets
from infera.server import metrics
from infera.server.profiling import fan_out_profile, select_targets

logger = logging.getLogger(__name__)

_REQUEST_ID_HEADER = "X-Infera-Request-Id"
# GAIE header-routing hint: the EPP writes the chosen worker id here; the
# direct router reads it off the body annotation below. See infera/gaie.
_WORKER_INSTANCE_HEADER = "x-worker-instance-id"
_PREFILL_INSTANCE_HEADER = "x-prefill-instance-id"


def _stash_direct_worker(body: dict, request: Request) -> None:
    """Copy the GAIE EPP's routing hints onto the body so DirectRouter can
    dispatch to the gateway-chosen worker(s): x-worker-instance-id (decode /
    primary) and, for disaggregated serving, x-prefill-instance-id. No-op when
    the headers are absent (auto mode / no gateway)."""
    from infera.router.direct import DIRECT_PREFILL_KEY, DIRECT_WORKER_KEY

    worker_id = request.headers.get(_WORKER_INSTANCE_HEADER)
    if worker_id:
        body[DIRECT_WORKER_KEY] = worker_id
        prefill_id = request.headers.get(_PREFILL_INSTANCE_HEADER)
        if prefill_id:
            body[DIRECT_PREFILL_KEY] = prefill_id


app = FastAPI(title="Infera")
registry: Registry | None = None
router: BaseRouter | None = None
kv_client: KvEventClient | None = None
# Issue #20 item 3 / PD §6.2 — socket path for the speculative-
# prefetch endpoint. None disables /v1/cache/prewarm (503). The
# endpoint opens a short-lived client per request: cheap (sub-ms
# UDS open+handshake) and avoids the asyncio.Lock-cross-loop trap
# a singleton client would hit when FastAPI's TestClient spins up
# a fresh loop per request.
_kvd_socket_path: str | None = None

# Profiling control plane. Disabled by default (mirrors dynamo's system status
# server being off unless DYN_SYSTEM_PORT is set). The httpx client used to
# fan profile control out to worker engine endpoints is created lazily on first
# use and kept separate from the router's client.
_enable_profiling: bool = False
_profile_client: httpx.AsyncClient | None = None


def init_app(
    reg: Registry,
    rtr: BaseRouter,
    kv: KvEventClient | None = None,
    kvd_socket_path: str | None = None,
    enable_profiling: bool = False,
) -> FastAPI:
    global registry, router, kv_client, _kvd_socket_path
    global _enable_profiling
    registry = reg
    router = rtr
    kv_client = kv
    _kvd_socket_path = kvd_socket_path
    _enable_profiling = enable_profiling
    return app


def _get_profile_client() -> httpx.AsyncClient:
    """Lazily build the shared httpx client for profile fan-out."""
    global _profile_client
    if _profile_client is None:
        _profile_client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0))
    return _profile_client


# ------------------------------------------------------------------
# Read-only worker inspection
# (Workers register themselves directly with etcd; the server is purely
# a reader of that state, so there are no write endpoints here.)
# ------------------------------------------------------------------


@app.get("/v1/admin/cache-view/{worker_id}")
async def cache_view(worker_id: str, dp_rank: int | None = None) -> dict:
    """Debug endpoint: inspect router-mirrored KV cache state for a worker
    (optionally a specific DP rank of a rank-multiplexed worker)."""
    if kv_client is None:
        raise HTTPException(status_code=503, detail="KV-aware mode disabled")
    view = kv_client.cache_view(worker_id, dp_rank)
    return {"worker_id": worker_id, "dp_rank": dp_rank, "block_count": len(view)}


# ------------------------------------------------------------------
# Unified torch-profiler control plane
# (Off unless --enable-profiling. Resolves target workers from the registry
# and forwards to each engine's native /start_profile|/stop_profile. See
# infera/server/profiling.py.)
# ------------------------------------------------------------------


async def _profile_action(action: str, request: Request) -> dict:
    """Shared handler for /v1/admin/profile/{start,stop}.

    Disabled -> 403. Unknown role -> 400. No matching worker -> 404. Otherwise
    fans the action out to every selected worker and returns the aggregate.
    Optional selectors come from the query string or JSON body: worker_id,
    model, role (mixed|prefill|decode); any other JSON body fields are passed
    through to the engine (e.g. SGLang's output_dir / num_steps).
    """
    if not _enable_profiling:
        raise HTTPException(
            status_code=403,
            detail="profiling disabled; start the server with --enable-profiling",
        )
    if registry is None:
        raise HTTPException(status_code=503, detail="registry not initialized")

    # Body is optional; tolerate empty / non-JSON bodies.
    try:
        body = await request.json()
    except (json.JSONDecodeError, ValueError):
        body = {}
    if not isinstance(body, dict):
        body = {}

    # Selectors can arrive via query params or body; query wins if both given.
    qp = request.query_params
    worker_id = qp.get("worker_id") or body.pop("worker_id", None)
    model = qp.get("model") or body.pop("model", None)
    role = qp.get("role") or body.pop("role", None)

    try:
        targets = select_targets(registry.list_all(), worker_id=worker_id, model=model, role=role)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not targets:
        raise HTTPException(
            status_code=404,
            detail="no active worker matched the given worker_id/model/role",
        )

    # Remaining body keys are forwarded verbatim to the engine endpoint.
    return await fan_out_profile(_get_profile_client(), targets, action, body)


@app.post("/v1/admin/profile/start")
async def profile_start(request: Request) -> dict:
    return await _profile_action("start", request)


@app.post("/v1/admin/profile/stop")
async def profile_stop(request: Request) -> dict:
    return await _profile_action("stop", request)


@app.get("/v1/workers")
async def list_workers() -> dict:
    workers = [
        {
            "worker_id": w.worker_id,
            "url": w.url,
            "model_name": w.model_name,
            "engine": w.engine,
            "status": w.status,
            "disagg_mode": w.disagg_mode,
            "disagg_meta": w.disagg_meta,
            "kv_events_endpoint": w.kv_events_endpoint,
            "kv_block_size": w.kv_block_size,
            "dp_rank": w.dp_rank,
            "dp_size": w.dp_size,
        }
        for w in registry.list_all()
    ]
    return {"workers": workers}


# OpenAI-compatible model listing. Used by sglang's bench_serving.py and other
# OpenAI clients to confirm the server is ready before sending requests. We
# expose every distinct model_name advertised by an active worker.
@app.get("/v1/models")
async def list_models() -> dict:
    import time as _time

    created = int(_time.time())
    seen: set[str] = set()
    data = []
    for w in registry.list_all():
        if w.status != "active" or not w.model_name or w.model_name in seen:
            continue
        seen.add(w.model_name)
        data.append(
            {
                "id": w.model_name,
                "object": "model",
                "created": created,
                "owned_by": "infera",
            }
        )
    return {"object": "list", "data": data}


# ------------------------------------------------------------------
# Inference endpoints
# ------------------------------------------------------------------


async def _dispatch_inference(request: Request, *, path: str) -> Any:
    """Shared body for /v1/chat/completions and /v1/completions.

    Both endpoints want the same request-id correlation + cache-hint
    parsing; they differ only in the path forwarded to the worker."""
    body = await request.json()
    # Generate or echo a request ID so router decisions can be correlated
    # across server + P/D worker logs. The router echoes it back to the
    # client in the response headers; cost-aware policies can include it
    # in their structured pick logs.
    request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex
    body["_infera_request_id"] = request_id
    # Parse cache_control once at the front door so the policy + engine
    # adapters can read it without re-walking the body. Anthropic-style
    # and OpenAI-style hints both supported (see infera/router/cache_control.py).
    from infera.router.cache_control import parse_cache_hints

    body["_infera_cache_hints"] = parse_cache_hints(body)
    # GAIE direct mode: honour the EPP's worker pick from the request header.
    _stash_direct_worker(body, request)
    resp = await router.dispatch(body, stream=bool(body.get("stream")), path=path)
    resp.headers[_REQUEST_ID_HEADER] = request_id
    return resp


@app.post("/v1/cache/prewarm")
async def cache_prewarm(request: Request) -> Response:
    """Speculative L3 prefetch hint (issue #20 item 3 / PD §6.2).

    Agentic harnesses call this with the block hashes they expect a
    follow-up request to load — e.g. after a tool call returns and
    before sending the next chat completion. The router forwards
    the hashes to kvd as a one-way `PrefetchHint`; kvd pulls them
    from L3 (long region / spillover) into the host RAM tier so
    the engine's subsequent `get` sees a fast hit instead of a cold
    miss.

    Request body::

        {
          "model": "MiniMax-M2.5",
          "block_hashes": ["<hex>", "<hex>", ...],
          "compat_key": "",           // optional, scopes per-engine
          "deadline_ms": 500          // optional, TTL on warmed entries
        }

    The harness is responsible for computing the block hashes (it
    received them from a prior engine response, or computed them
    with the engine's hash function under a deterministic
    PYTHONHASHSEED). We do not derive them here — the router would
    have to duplicate the engine's hashing logic, which couples us
    to a specific engine version.

    Returns 202 Accepted immediately (no waiting on kvd worker).
    Returns 503 if kvd isn't reachable (operator didn't set
    `kvd_socket_path` or the daemon is down)."""
    try:
        body = await request.json()
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"bad json: {exc}") from exc

    model = body.get("model")
    if not isinstance(model, str) or not model:
        raise HTTPException(status_code=400, detail="`model` is required")
    hex_hashes = body.get("block_hashes")
    if not isinstance(hex_hashes, list) or not all(isinstance(h, str) for h in hex_hashes):
        raise HTTPException(status_code=400, detail="`block_hashes` must be a list of hex strings")
    if not hex_hashes:
        # Empty list is a no-op but legal — 202 to match agentic-client
        # ergonomics (a harness that built an empty list shouldn't get
        # a 4xx).
        return Response(
            content=json.dumps({"accepted": True, "n_keys": 0}),
            status_code=202,
            media_type="application/json",
        )
    try:
        keys = [bytes.fromhex(h) for h in hex_hashes]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"bad hex in block_hashes: {exc}") from exc

    compat_key = body.get("compat_key", "")
    if not isinstance(compat_key, str):
        raise HTTPException(status_code=400, detail="`compat_key` must be a string")
    deadline_ms_raw = body.get("deadline_ms", 1000)
    try:
        deadline_ms = int(deadline_ms_raw)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"`deadline_ms`: {exc}") from exc
    if deadline_ms <= 0:
        raise HTTPException(status_code=400, detail="`deadline_ms` must be positive")

    if _kvd_socket_path is None:
        raise HTTPException(
            status_code=503,
            detail="prefetch unavailable — kvd_socket_path not configured",
        )

    # Short-lived client per request — see the comment on
    # `_kvd_socket_path` for why we don't memoize. The whole flow
    # (open + hello + hint + close) is sub-millisecond on UDS.
    try:
        async with KvdClient(_kvd_socket_path, client_id="infera-router-prefetch") as client:
            await client.prefetch_hint(
                keys, model=model, compat_key=compat_key, deadline_ms=deadline_ms
            )
    except KvdConnectionError as exc:
        raise HTTPException(status_code=503, detail=f"kvd unreachable: {exc}") from exc

    return Response(
        content=json.dumps({"accepted": True, "n_keys": len(keys)}),
        status_code=202,
        media_type="application/json",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    return await _dispatch_inference(request, path="/v1/chat/completions")


@app.post("/v1/completions")
async def completions(request: Request) -> Any:
    return await _dispatch_inference(request, path="/v1/completions")


# ------------------------------------------------------------------
# Anthropic Messages API (`/v1/messages`)
# ------------------------------------------------------------------
# Lets clients like openclaw point ANTHROPIC_BASE_URL at Infera and
# talk to the same SGLang/vLLM workers that back the OpenAI endpoints.
# Body shape gets translated to OpenAI Chat before dispatch; response
# is translated back to Anthropic SSE / JSON. cache_control passes
# through `parse_cache_hints` unchanged. See infera/api/anthropic.py.


@app.post("/v1/messages")
async def anthropic_messages(request: Request) -> Any:
    from fastapi.responses import JSONResponse, StreamingResponse

    from infera.api.anthropic import (
        AnthropicRequestRejected,
        anthropic_to_openai_request,
        openai_to_anthropic_response,
        openai_to_anthropic_sse,
    )
    from infera.router.cache_control import parse_cache_hints

    # `anthropic-version` is required by the Anthropic spec. We accept
    # any value but log unsupported ones so operators can see if a
    # client is using a version we haven't tested against. Note: the
    # value is only LOGGED — translation behavior doesn't gate on it.
    # Documented in the module docstring.
    anthropic_version = request.headers.get("anthropic-version", "")
    if anthropic_version and anthropic_version != "2023-06-01":
        logger.info(
            "anthropic-version=%s — accepting (only 2023-06-01 is regression-tested in this build; "
            "value is logged but does not gate translation behavior)",
            anthropic_version,
        )
    # Auth headers — accepted without validation in v1. Production
    # deployments front this with their own gateway. Log at DEBUG so
    # operators can confirm "auth is passthrough" by inspecting logs.
    if request.headers.get("x-api-key") or request.headers.get("authorization"):
        logger.debug(
            "anthropic-messages: auth header present but not validated in v1 "
            "(deploy behind a gateway for auth enforcement)"
        )

    try:
        anthropic_body = await request.json()
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    # Reject missing `model` — Anthropic requires it; without it we'd
    # echo back `"model": ""` in the translated response which is
    # invalid Anthropic shape. PR #13 review fix P1.
    model = anthropic_body.get("model") if isinstance(anthropic_body, dict) else None
    if not isinstance(model, str) or not model.strip():
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "missing or empty 'model' field (required by Anthropic Messages spec)",
                },
            },
        )

    # Parse cache hints from the Anthropic body BEFORE translating.
    # parse_cache_hints already understands the Anthropic shape
    # (system[].cache_control, tools[].cache_control, messages[].content[].cache_control).
    cache_hints = parse_cache_hints(anthropic_body)
    request_id = request.headers.get(_REQUEST_ID_HEADER) or uuid.uuid4().hex

    # Translate to OpenAI shape. Refuses tools/MM with explicit 400.
    try:
        openai_body = anthropic_to_openai_request(anthropic_body)
    except AnthropicRequestRejected as exc:
        return JSONResponse(
            status_code=400,
            content={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": str(exc)},
            },
        )

    # Stash the same internal annotations the OpenAI endpoints use so
    # downstream machinery (cache_hints injection, request id
    # correlation) keeps working without special-casing.
    openai_body["_infera_request_id"] = request_id
    openai_body["_infera_cache_hints"] = cache_hints
    _stash_direct_worker(openai_body, request)
    stream = bool(anthropic_body.get("stream", False))

    # When streaming, ask the engine to include usage in the final
    # chunk. Most engines (vLLM default, SGLang without this) emit
    # `output_tokens: 0` in their streaming chunks — without
    # include_usage we'd report `output_tokens: 0` on every Anthropic
    # message_delta. PR #13 review fix P0-2.
    if stream:
        existing = openai_body.get("stream_options")
        if not isinstance(existing, dict):
            existing = {}
        if "include_usage" not in existing:
            existing = {**existing, "include_usage": True}
        openai_body["stream_options"] = existing

    # Dispatch the OpenAI-shaped request to the worker.
    engine_resp = await router.dispatch(openai_body, stream=stream, path="/v1/chat/completions")

    # If the engine / router returned an error (4xx/5xx), forward it
    # verbatim instead of translating an empty assistant message —
    # silent error masking would otherwise hide upstream failures
    # from clients. PR #13 review fix P0-1.
    if engine_resp.status_code >= 400:
        try:
            engine_resp.headers[_REQUEST_ID_HEADER] = request_id
        except Exception:
            pass
        return engine_resp

    # Translate the response back to Anthropic shape.
    if stream:
        # engine_resp is a StreamingResponse holding OpenAI SSE
        # bytes; wrap its iterator. Use the validated `model` string.
        async def _anthropic_stream():
            async for chunk in openai_to_anthropic_sse(
                engine_resp.body_iterator, model=model, msg_id_hint=request_id
            ):
                yield chunk

        out = StreamingResponse(_anthropic_stream(), media_type="text/event-stream")
        out.headers[_REQUEST_ID_HEADER] = request_id
        # Preserve content-type for SSE clients that key off it.
        return out

    # Non-streaming: engine_resp.body is bytes of the OpenAI JSON
    # response (FastAPI's JSONResponse stores it pre-encoded).
    try:
        openai_payload = json.loads(engine_resp.body)
    except Exception:
        # Forward upstream errors unchanged — engine returned non-JSON.
        return engine_resp

    anthropic_payload = openai_to_anthropic_response(
        openai_payload, model=model, msg_id_hint=request_id
    )
    out = JSONResponse(content=anthropic_payload, status_code=engine_resp.status_code)
    out.headers[_REQUEST_ID_HEADER] = request_id
    return out


@app.get("/health")
async def health() -> dict:
    workers = registry.pool.list_active()
    return {"status": "ok", "active_workers": len(workers)}


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    """Prometheus text-exposition endpoint. Snapshot the worker-pool
    gauges on every scrape so they stay live without per-event updates.
    """
    if registry is not None:
        # Reset gauges then re-populate to handle workers leaving the fleet.
        metrics.active_workers.clear()
        by_mode: dict[str, int] = {}
        for w in registry.list_all():
            if w.status != WorkerStatus.ACTIVE:
                continue
            key = (
                w.disagg_mode.value if isinstance(w.disagg_mode, DisaggMode) else str(w.disagg_mode)
            )
            by_mode[key] = by_mode.get(key, 0) + 1
        for mode, n in by_mode.items():
            metrics.active_workers.labels(disagg_mode=mode).set(n)

    if kv_client is not None:
        metrics.policy_cache_view_size.clear()
        workers = registry.list_all() if registry is not None else []
        for t in expand_targets(workers):
            view = kv_client.cache_view(t.worker.worker_id, t.dp_rank)
            metrics.policy_cache_view_size.labels(worker_id=t.route_key).set(len(view))

    body, content_type = metrics.render_metrics()
    return Response(content=body, media_type=content_type)
