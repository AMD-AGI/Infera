###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import DisaggMode
from infera.router.base import BaseRouter
from infera.router.cache_control import parse_cache_hints
from infera.router.dp_routing import dp_rank_header
from infera.router.engine_priority import inject_engine_priority
from infera.server import metrics

logger = logging.getLogger(__name__)


class _Retry(Exception):
    """Internal signal: a dispatch attempt failed BEFORE any response data was
    sent to the client, so the router may transparently fail over to another
    worker. Carries the error Response to return if retries are exhausted."""

    def __init__(self, response: Response) -> None:
        self.response = response


class MixedRouter(BaseRouter):
    """Plain forward router for mixed (non-PD) workers.

    Picks one worker via the policy and proxies the request as-is. If a
    dispatch fails before any response data has reached the client (worker
    unreachable / NATS error / idle-timeout-before-first-token / 429 backlog),
    it transparently re-selects another worker, up to ``request_max_retries``
    times. A failure AFTER the first chunk has been streamed is never retried
    (the client already holds partial output).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Bound connect time so unreachable workers fail fast; leave read open
        # for arbitrarily long generations. Bump connection limits well above
        # httpx defaults (100) so we can sustain high-concurrency benchmarks
        # (e.g. 4096 in-flight requests per worker URL).
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=60.0),
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=4096,
            ),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def dispatch(
        self,
        body: dict,
        *,
        stream: bool,
        path: str = "/v1/chat/completions",
    ) -> Response:
        with metrics.track_request(router="mixed") as obs:
            model = body.get("model")
            # cache_control hints are body-level, computed once per request.
            hints = body.get("_infera_cache_hints") or parse_cache_hints(body)

            tried: set[str] = set()
            last_error: Response | None = None
            # 1 initial attempt + up to request_max_retries failovers.
            for _ in range(1 + self.request_max_retries):
                candidates = [
                    w
                    for w in self.pool.list_active(model=model, mode=DisaggMode.MIXED)
                    if w.worker_id not in tried
                ]
                if not candidates:
                    break
                target, blocks = self.policy.pick(candidates, body)
                tried.add(target.worker.worker_id)
                try:
                    return await self._attempt(target, blocks, body, hints, path, stream, obs)
                except _Retry as r:
                    last_error = r.response
                    logger.info(
                        "failover: worker %s failed before first byte; %d worker(s) tried",
                        target.worker.worker_id,
                        len(tried),
                    )
                    continue

            if last_error is not None:
                return last_error
            obs["outcome"] = "503"
            return JSONResponse(
                content={"error": f"no active mixed worker for model={model!r}"},
                status_code=503,
            )

    async def _attempt(self, target, blocks, body, hints, path, stream, obs) -> Response:
        """One dispatch attempt to ``target``. Returns a committed Response, or
        raises :class:`_Retry` when it fails before any client data was sent."""
        worker = target.worker
        url = f"{worker.url}{path}"
        dp_headers = dp_rank_header(target)

        # Engine-specific priority injection depends on the chosen worker.
        forwarded_body = inject_engine_priority(body, hints, worker.engine)
        forwarded_body.pop("_infera_cache_hints", None)
        forwarded_body.pop("_infera_request_id", None)

        use_nats = self.nats_client is not None and worker.request_transport == "nats"
        self.policy.on_request_started(target.route_key, blocks)

        # Optional NATS admission throttle: a worker at its backlog limit is a
        # pre-first-byte failure -> retryable to spread to a freer worker.
        if use_nats and not await self.nats_client.admit(worker.worker_id):
            self.policy.on_request_finished(target.route_key, blocks)
            obs["outcome"] = "429"
            raise _Retry(
                JSONResponse(
                    content={"error": f"worker {worker.worker_id} request backlog over limit"},
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            )

        if stream:
            return await self._attempt_stream(
                target, blocks, worker, url, forwarded_body, dp_headers, path, use_nats, obs
            )
        try:
            return await self._attempt_unary(
                worker, url, forwarded_body, dp_headers, path, use_nats, obs
            )
        finally:
            self.policy.on_request_finished(target.route_key, blocks)

    async def _attempt_stream(
        self, target, blocks, worker, url, forwarded_body, dp_headers, path, use_nats, obs
    ) -> Response:
        """Peek the first reply event: if it's data, commit and stream it +
        the rest; if it's an error/empty before any data, raise _Retry."""
        agen = self._normalized_stream(worker, url, forwarded_body, dp_headers, path, use_nats)
        try:
            kind, status, data0 = await agen.__anext__()
        except StopAsyncIteration:
            kind, status, data0 = TYPE_DONE, 200, b""

        if kind == TYPE_DATA:
            obs["outcome"] = "ok"  # committed once first byte is in hand

            async def generate():
                try:
                    if data0:
                        yield data0
                    async for k, _st, d in agen:
                        if k == TYPE_DATA:
                            if d:
                                yield d
                        elif k == TYPE_ERROR:
                            logger.warning(
                                "stream from worker %s failed mid-stream: %s",
                                worker.worker_id,
                                d[:200],
                            )
                            yield (
                                f'data: {{"error":"worker {worker.worker_id} '
                                f'stream failed mid-stream"}}\n\n'
                            ).encode()
                            return
                        else:  # done
                            return
                finally:
                    self.policy.on_request_finished(target.route_key, blocks)

            return StreamingResponse(generate(), media_type="text/event-stream")

        # Failure before any data -> retryable. Close the generator, undo the
        # policy bookkeeping for this attempt, and signal failover.
        await agen.aclose()
        self.policy.on_request_finished(target.route_key, blocks)
        code = status if (status and status >= 400) else 502
        obs["outcome"] = str(code)
        raise _Retry(
            JSONResponse(
                content={
                    "error": f"worker {worker.worker_id} failed before first token",
                    "raw": data0[:500].decode("utf-8", "replace") if data0 else "",
                },
                status_code=code,
            )
        )

    async def _attempt_unary(
        self, worker, url, forwarded_body, dp_headers, path, use_nats, obs
    ) -> Response:
        """Non-streaming attempt. Returns the worker's JSON response; raises
        _Retry on a transport-level failure (nothing sent to the client yet)."""
        if use_nats:
            payload = {"path": path, "stream": False, "headers": dp_headers, "body": forwarded_body}
            chunks: list[bytes] = []
            status = 200
            async for kind, st, data in self.nats_client.stream(worker.worker_id, payload):
                if kind == TYPE_DATA:
                    chunks.append(data)
                elif kind == TYPE_ERROR:
                    code = st or 502
                    obs["outcome"] = str(code)
                    raise _Retry(
                        JSONResponse(
                            content={
                                "error": f"worker {worker.worker_id} nats failed",
                                "raw": data[:500].decode("utf-8", "replace"),
                            },
                            status_code=code,
                        )
                    )
                else:  # done
                    status = st or 200
                    break
            raw = b"".join(chunks)
            try:
                payload_json = json.loads(raw) if raw else {}
            except ValueError:
                obs["outcome"] = "502"
                raise _Retry(
                    JSONResponse(
                        content={
                            "error": f"worker {worker.worker_id} returned non-JSON over nats",
                            "raw": raw[:500].decode("utf-8", "replace"),
                        },
                        status_code=502,
                    )
                ) from None
            obs["outcome"] = "ok" if status < 400 else f"{status // 100}xx"
            return JSONResponse(content=payload_json, status_code=status)

        # Direct HTTP forward.
        try:
            resp = await self._client.post(url, json=forwarded_body, headers=dp_headers)
        except httpx.HTTPError as exc:
            obs["outcome"] = "502"
            raise _Retry(
                JSONResponse(
                    content={"error": f"worker {worker.worker_id} unreachable: {exc}"},
                    status_code=502,
                )
            ) from exc
        try:
            payload_json = resp.json()
        except ValueError:
            obs["outcome"] = "502"
            raise _Retry(
                JSONResponse(
                    content={
                        "error": f"worker {worker.worker_id} returned non-JSON",
                        "raw": resp.text[:500],
                    },
                    status_code=502,
                )
            ) from None
        obs["outcome"] = "ok" if resp.status_code < 400 else f"{resp.status_code // 100}xx"
        return JSONResponse(content=payload_json, status_code=resp.status_code)

    async def _normalized_stream(
        self, worker, url, forwarded_body, dp_headers, path, use_nats
    ) -> AsyncIterator[tuple]:
        """Unify HTTP and NATS streaming into ``(kind, status, data)`` events
        where ``kind`` is one of TYPE_DATA / TYPE_DONE / TYPE_ERROR."""
        if use_nats:
            payload = {"path": path, "stream": True, "headers": dp_headers, "body": forwarded_body}
            async for kind, st, data in self.nats_client.stream(worker.worker_id, payload):
                yield (kind, st, data)
            return

        try:
            async with self._client.stream(
                "POST", url, json=forwarded_body, headers=dp_headers
            ) as resp:
                if resp.status_code >= 400:
                    err = await resp.aread()
                    yield (TYPE_ERROR, resp.status_code, err)
                    return
                async for chunk in resp.aiter_raw():
                    if chunk:
                        yield (TYPE_DATA, None, chunk)
            yield (TYPE_DONE, 200, b"")
        except httpx.HTTPError as exc:
            yield (TYPE_ERROR, None, str(exc).encode())
