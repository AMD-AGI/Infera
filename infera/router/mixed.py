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

from infera.common.nats_request import (
    DRAINING_NOTICE,
    TYPE_DATA,
    TYPE_DONE,
    TYPE_ERROR,
)
from infera.common.worker_pool import DisaggMode
from infera.router.base import BaseRouter
from infera.router.breaker import is_worker_fault
from infera.router.cache_control import parse_cache_hints
from infera.router.dp_routing import dp_rank_header
from infera.router.engine_priority import inject_engine_priority
from infera.router.migration import MigrationState, as_chat_chunk
from infera.router.token_ids import supports_streaming_ids
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

    def __init__(self, *args, migration_limit: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # How many times one generation may be carried to another worker. Zero
        # disables it; see infera.router.migration for what carrying costs.
        self._migration_limit = max(0, int(migration_limit or 0))
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
                # Drop workers the breaker has open. Falls back to the unfiltered
                # list when every candidate is open -- a request served by a
                # probably-bad worker beats turning a partial outage into a 503.
                candidates = self.breaker.filter(candidates)
                if not candidates:
                    break
                target, blocks = self.policy.pick(candidates, body)
                tried.add(target.worker.worker_id)
                try:
                    resp = await self._attempt(target, blocks, body, hints, path, stream, obs)
                    # Only a clean response is evidence the worker is healthy.
                    # A 4xx says the request was bad -- every worker would
                    # answer the same -- so scoring it as recovery would reset
                    # the failure count and reopen a breaker that should stay
                    # shut. It still has to free the probe slot it took.
                    if getattr(resp, "status_code", 200) < 400:
                        self.breaker.record_success(target.worker.worker_id)
                    else:
                        self.breaker.record_neutral(target.worker.worker_id)
                    return resp
                except _Retry as r:
                    # Pre-first-byte only: _Retry is never raised once bytes have
                    # been streamed, so a mid-stream failure cannot trip this.
                    # 4xx is retried but not held against the worker -- see
                    # is_worker_fault().
                    if is_worker_fault(getattr(r.response, "status_code", 0)):
                        self.breaker.record_failure(target.worker.worker_id)
                    else:
                        self.breaker.record_neutral(target.worker.worker_id)
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

        # Ask for the sampled token ids only when a migration could actually use
        # them: they cost the engine work and the client never sees them, since
        # they are taken back out before the stream is forwarded.
        if (
            stream
            and self._migration_limit > 0
            and supports_streaming_ids(worker.engine, path)
            and "return_token_ids" not in forwarded_body
        ):
            forwarded_body["return_token_ids"] = True

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

            state = self._migration_state(forwarded_body, use_nats, path)

            def passthrough(raw):
                # observe() also takes the router-only token ids back out, so
                # what is yielded is always what the caller asked for.
                return state.observe(raw) if state is not None else raw

            def to_client_shape(raw):
                return as_chat_chunk(passthrough(raw))

            async def generate():
                current, cur_target, cur_blocks = agen, target, blocks
                transform = passthrough
                try:
                    if data0:
                        yield transform(data0)
                    while True:
                        async for k, _st, d in current:
                            if k == TYPE_DATA:
                                if d:
                                    yield transform(d)
                            elif k == TYPE_ERROR:
                                # A drain notice is a planned handover, not a
                                # fault: the worker is leaving and expects us to
                                # take the generation elsewhere.
                                draining = d.startswith(DRAINING_NOTICE)
                                if draining:
                                    reason = "worker_draining"
                                    logger.info(
                                        "worker %s is draining; taking its stream elsewhere",
                                        cur_target.worker.worker_id,
                                    )
                                else:
                                    reason = "stream_broken"
                                    logger.warning(
                                        "stream from worker %s failed mid-stream: %s",
                                        cur_target.worker.worker_id,
                                        d[:200],
                                    )
                                break
                            else:  # done
                                return
                        else:
                            # The generator ended without saying why, which is
                            # the shape a crashed worker leaves behind.
                            reason = "stream_broken"

                        # The stream broke. Whatever the client already read has
                        # to be honoured, so the only options are to carry the
                        # generation to another worker or to end it visibly.
                        resumed = None
                        if state is not None and state.can_migrate():
                            resumed = await self._resume_elsewhere(
                                state, cur_target, path, obs, reason
                            )
                        elif state is not None:
                            # Migration was possible for this request and is not
                            # any more, which is worth counting separately from
                            # a deployment that never enabled it.
                            metrics.migrations_failed_total.labels(
                                reason="poisoned" if state.poisoned else "limit"
                            ).inc()
                        if resumed is None:
                            what = "is shutting down" if reason == "worker_draining" else "failed"
                            yield (
                                f'data: {{"error":"worker {cur_target.worker.worker_id} '
                                f'{what} mid-stream"}}\n\n'
                            ).encode()
                            return
                        current, next_target, next_blocks, reshape = resumed
                        transform = to_client_shape if reshape else passthrough
                        # The previous attempt's load accounting ends here; the
                        # new one is already counted by _resume_elsewhere.
                        self.policy.on_request_finished(cur_target.route_key, cur_blocks)
                        cur_target, cur_blocks = next_target, next_blocks
                finally:
                    self.policy.on_request_finished(cur_target.route_key, cur_blocks)

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

    def _migration_state(self, forwarded_body, use_nats, path):
        """State for carrying this generation elsewhere, or None if it cannot be.

        Only over NATS: on the HTTP path the router hands the connection to the
        engine and never sees a frame, so there is nothing to accumulate and
        nothing to resume from.
        """
        if not use_nats or self._migration_limit <= 0:
            return None
        return MigrationState(forwarded_body, limit=self._migration_limit, path=path)

    async def _resume_elsewhere(self, state, failed_target, path, obs, reason):
        """Continue this generation on a different worker.

        Returns the new event stream and its target, or None when nobody else
        can take it -- in which case the caller ends the stream visibly rather
        than leaving the client waiting on a generation that stopped.
        """
        cont = state.next_continuation()
        body = cont.body
        # An exact chat continuation is issued against the completions endpoint,
        # so what is sent and what the client is reading can differ.
        send_path, client_path = cont.path, path
        model = body.get("model")
        candidates = [
            w
            for w in self.pool.list_active(model=model, mode=DisaggMode.MIXED)
            if w.worker_id != failed_target.worker.worker_id
        ]
        # The failed worker is excluded even if it is the only one: it just
        # dropped this stream, and handing the request back to it is how a
        # migration loop starts.
        candidates = self.breaker.filter(candidates)
        if not candidates:
            logger.warning("cannot migrate: no other worker serves model=%r", model)
            metrics.migrations_failed_total.labels(reason="no_candidate").inc()
            return None

        target, blocks = self.policy.pick(candidates, body)
        worker = target.worker
        if worker.request_transport != "nats" or self.nats_client is None:
            # The accumulated state is only resumable over NATS.
            logger.warning("cannot migrate: worker %s is not on nats", worker.worker_id)
            metrics.migrations_failed_total.labels(reason="not_nats").inc()
            return None

        self.policy.on_request_started(target.route_key, blocks)
        agen = self._normalized_stream(
            worker,
            f"{worker.url}{send_path}",
            body,
            dp_rank_header(target),
            send_path,
            use_nats=True,
        )
        # Peek: a worker that cannot take it must not consume the migration
        # budget silently, and the caller needs a stream that is already
        # producing before it commits to it.
        try:
            kind, _st, first = await agen.__anext__()
        except StopAsyncIteration:
            kind, first = TYPE_DONE, b""
        if kind != TYPE_DATA:
            await agen.aclose()
            self.policy.on_request_finished(target.route_key, blocks)
            logger.warning("migration to %s failed before first byte", worker.worker_id)
            metrics.migrations_failed_total.labels(reason="no_first_byte").inc()
            return None

        logger.info(
            "migrated a live generation from %s to %s after %d token(s) (%s)",
            failed_target.worker.worker_id,
            worker.worker_id,
            state.produced_tokens,
            "exact token ids" if cont.exact else "carried text",
        )
        metrics.migrations_total.labels(reason=reason).inc()
        obs["outcome"] = "ok"

        async def resumed():
            # The peeked frame is replayed unobserved: the caller runs every
            # frame through the same transform, and recording it here as well
            # would count its tokens twice.
            if first:
                yield TYPE_DATA, None, first
            async for item in agen:
                yield item

        # Replies arrive in the shape of whatever endpoint was used, which is
        # not necessarily the one the client is reading.
        reshape = send_path != client_path
        return resumed(), target, blocks, reshape

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
            # Same rule as the HTTP path below: a 5xx before any data is a
            # worker fault and retryable; a 4xx belongs to the request.
            if is_worker_fault(status):
                raise _Retry(JSONResponse(content=payload_json, status_code=status))
            return JSONResponse(content=payload_json, status_code=status)

        # Direct HTTP forward.
        try:
            resp = await self._client.post(url, json=forwarded_body, headers=dp_headers)
        except httpx.HTTPError as exc:
            obs["outcome"] = "502"
            logger.warning(
                "worker %s unreachable (%s: %s)",
                worker.worker_id,
                type(exc).__name__,
                exc,
            )
            raise _Retry(
                JSONResponse(
                    content={"error": f"worker {worker.worker_id} unreachable"},
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
        # A 5xx here is the worker failing before a single byte reached the
        # client, which is exactly the case failover exists for -- and until now
        # this path returned it verbatim instead, so a unary request over HTTP
        # never failed over and never fed the circuit breaker. The streaming
        # path and the Rust router both retry it; this brings the third one into
        # line. 4xx still passes straight through: the request itself is bad and
        # every worker would say the same, so retrying only triples the latency
        # of an error the client needs to see.
        if is_worker_fault(resp.status_code):
            raise _Retry(JSONResponse(content=payload_json, status_code=resp.status_code))
        return JSONResponse(content=payload_json, status_code=resp.status_code)

    async def _normalized_stream(
        self, worker, url, forwarded_body, dp_headers, path, use_nats
    ) -> AsyncIterator[tuple]:
        """Unify HTTP and NATS streaming into ``(kind, status, data)`` events
        where ``kind`` is one of TYPE_DATA / TYPE_DONE / TYPE_ERROR."""
        if use_nats:
            payload = {
                "path": path,
                "stream": True,
                "headers": dp_headers,
                "body": forwarded_body,
                # Lets a draining worker return this stream immediately instead
                # of holding its shutdown open; only true when we can resume it.
                "migratable": self._migration_limit > 0,
            }
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
            logger.warning(
                "stream from worker %s failed before first byte: %s: %s",
                worker.worker_id,
                type(exc).__name__,
                exc,
            )
            yield (TYPE_ERROR, None, b"worker stream transport error")
