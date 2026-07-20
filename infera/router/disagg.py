###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio
import json
import logging
import random

import httpx
from fastapi import Response
from fastapi.responses import JSONResponse, StreamingResponse

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import DisaggMode
from infera.router.base import BaseRouter
from infera.router.cache_control import parse_cache_hints
from infera.router.disagg_protocols import (
    ProtocolMismatch,
    UnknownProtocol,
    resolve_protocol,
)
from infera.router.dp_routing import (
    align_room_to_prefill_rank,
    dp_rank_header,
    inject_disagg_prefill_dp_rank,
)
from infera.router.engine_priority import inject_engine_priority
from infera.router.policy.target import RouteTarget
from infera.server import metrics

logger = logging.getLogger(__name__)


def _generate_room_id() -> int:
    """Random u63 ID for the per-request session (SGLang's bootstrap_room,
    vLLM connectors' transfer_id)."""
    return random.randrange(2**63)


def _leg_headers(forged_id: str | None, target: RouteTarget) -> dict[str, str] | None:
    """Shared forged request id (if any) + this leg's DP-rank pin."""
    headers: dict[str, str] = {}
    if forged_id:
        headers["X-Request-Id"] = forged_id
    headers.update(dp_rank_header(target) or {})
    return headers or None


class DisaggRouter(BaseRouter):
    """Dual-dispatch router for PD-disaggregated workers.

    Delegates body shaping to a ``DisaggProtocol`` resolved from the
    workers' ``disagg_meta["protocol"]`` tag. Router owns connection
    lifecycle, retries, and metrics; protocols are pure functions over
    request bodies.

    Two topologies, picked from ``proto.topology``:
      - ``concurrent``: POST P and D in parallel, stream D back, drain
        P in the background (SGLang, vLLM-mooncake).
      - ``serial-pull``: POST P, await its response, extract handoff
        fields, then POST D (vLLM-mori-read, vLLM-nixl).
    """

    # Pre-flight decode POST is idempotent (engine hasn't parsed the
    # body yet, otherwise it would have started responding), so retry
    # on transport errors. Never retry after any chunk has arrived —
    # engine has committed work and a retry would double-bill.
    _DECODE_OPEN_MAX_RETRIES = 3
    _DECODE_OPEN_INITIAL_BACKOFF_S = 0.05
    _DECODE_OPEN_MAX_BACKOFF_S = 0.5

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # No connection cap: each request holds two long-lived streams
        # (P + D), so a cap below 2*concurrency deadlocks. Keep-alive off:
        # reusing an idle connection raced engine-side timeout_keep_alive=5
        # — write onto a half-closed socket, decode never lands, but prefill
        # already registered the bootstrap_room → decode hangs on KVPoll for
        # the 300s mooncake timeout.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(None, connect=60.0),
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=0,
            ),
        )
        # Strong refs to in-flight prefill POSTs: create_task only holds a
        # weak ref, so a GC'd task aborts the half-sent request
        # (KVTransferError → decode hangs on KVPoll 300s).
        self._pending_prefill_tasks: set[asyncio.Task] = set()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def dispatch(
        self,
        body: dict,
        *,
        stream: bool,
        path: str = "/v1/chat/completions",
    ) -> Response:
        with metrics.track_request(router="disagg") as obs:
            model = body.get("model")
            prefills = self.pool.list_active(model=model, mode=DisaggMode.PREFILL)
            decodes = self.pool.list_active(model=model, mode=DisaggMode.DECODE)
            if not prefills or not decodes:
                obs["outcome"] = "503"
                metrics.pd_bootstrap_failures_total.labels(reason="no_pd_workers").inc()
                return JSONResponse(
                    content={"error": f"need both prefill and decode workers for model={model!r}"},
                    status_code=503,
                )

            # role_hint lets cost-aware policies weight P (cache-heavy: a hit
            # skips prefill) vs D (load-heavy) differently.
            p_target, p_blocks = self.policy.pick(prefills, body, role_hint="prefill")
            d_target, d_blocks = self.policy.pick(decodes, body, role_hint="decode")
            return await self._run_pd(
                obs, p_target, d_target, p_blocks, d_blocks, body, stream, path
            )

    async def dispatch_direct(
        self,
        body: dict,
        *,
        stream: bool,
        path: str,
        prefill_id: str,
        decode_id: str,
    ) -> Response:
        """PD dispatch with workers already chosen upstream (GAIE EPP direct
        mode). Looks the prefill/decode workers up by id and runs the same
        protocol/topology/transport machinery as :meth:`dispatch`, but skips
        ``policy.pick`` — selection happened in the EPP. Empty block lists make
        the policy in-flight refcounting a no-op (the EPP owns bookkeeping)."""
        with metrics.track_request(router="disagg") as obs:
            p = self.pool.get(prefill_id)
            d = self.pool.get(decode_id)
            if p is None or d is None:
                obs["outcome"] = "503"
                metrics.pd_bootstrap_failures_total.labels(reason="direct_worker_missing").inc()
                missing = prefill_id if p is None else decode_id
                return JSONResponse(
                    content={
                        "error": f"PD worker {missing!r} from gateway not found (stale routing?)"
                    },
                    status_code=503,
                )
            return await self._run_pd(
                obs, RouteTarget(p), RouteTarget(d), [], [], body, stream, path
            )

    async def _run_pd(
        self,
        obs,
        p_target: RouteTarget,
        d_target: RouteTarget,
        p_blocks: list[int],
        d_blocks: list[int],
        body: dict,
        stream: bool,
        path: str,
    ) -> Response:
        """Run the PD dual-dispatch for already-selected prefill/decode targets:
        resolve the protocol, forge the bootstrap room/request id, and hand off
        to the concurrent or serial-pull dispatcher. Shared by the policy-driven
        :meth:`dispatch` and the gateway-driven :meth:`dispatch_direct`."""
        p, d = p_target.worker, d_target.worker
        try:
            proto = resolve_protocol(p, d)
        except (ProtocolMismatch, UnknownProtocol) as exc:
            obs["outcome"] = "500"
            metrics.pd_bootstrap_failures_total.labels(reason="protocol_unresolved").inc()
            return JSONResponse(content={"error": str(exc)}, status_code=500)

        # SGLang's follow_bootstrap_room balancer ties the prefill DP rank to
        # bootstrap_room % dp_size; encode the steered rank so the prefill
        # sender's consistency check passes (no engine env var needed).
        room_id = align_room_to_prefill_rank(_generate_room_id(), p_target)

        base = dict(body)
        hints = base.pop("_infera_cache_hints", None) or parse_cache_hints(body)
        base.pop("_infera_request_id", None)
        base.pop("_infera_direct_worker", None)
        base.pop("_infera_direct_prefill", None)

        p_url = f"{p.url}{path}"
        d_url = f"{d.url}{path}"

        # request_id_for may raise (e.g. malformed disagg_meta); compute before
        # on_request_started so no started/finished bookkeeping is needed on
        # the failure path.
        try:
            forged_id = proto.request_id_for(p, d, room_id)
        except ValueError as exc:
            obs["outcome"] = "500"
            metrics.pd_bootstrap_failures_total.labels(reason="protocol_request_id_failed").inc()
            return JSONResponse(content={"error": str(exc)}, status_code=500)

        self.policy.on_request_started(p_target.route_key, p_blocks)
        self.policy.on_request_started(d_target.route_key, d_blocks)

        dispatcher = (
            self._dispatch_concurrent if proto.topology == "concurrent" else self._dispatch_serial
        )
        return await dispatcher(
            obs,
            proto,
            base,
            hints,
            p_target,
            p_blocks,
            p_url,
            d_target,
            d_blocks,
            d_url,
            room_id,
            stream,
            forged_id,
        )

    async def _dispatch_concurrent(
        self,
        obs,
        proto,
        base,
        hints,
        p_target,
        p_blocks,
        p_url,
        d_target,
        d_blocks,
        d_url,
        room_id,
        stream,
        forged_id: str | None,
    ) -> Response:
        p, d = p_target.worker, d_target.worker
        p_headers = _leg_headers(forged_id, p_target)
        d_headers = _leg_headers(forged_id, d_target)
        try:
            p_body = inject_engine_priority(
                proto.annotate_prefill(base, p, d, room_id), hints, p.engine
            )
            d_body = inject_disagg_prefill_dp_rank(
                inject_engine_priority(
                    proto.annotate_decode(base, p, d, room_id, None), hints, d.engine
                ),
                prefill_target=p_target,
                decode_engine=d.engine,
            )
        except ValueError as exc:
            self.policy.on_request_finished(p_target.route_key, p_blocks)
            self.policy.on_request_finished(d_target.route_key, d_blocks)
            obs["outcome"] = "500"
            metrics.pd_bootstrap_failures_total.labels(reason="protocol_annotate_failed").inc()
            return JSONResponse(content={"error": str(exc)}, status_code=500)

        # Deliver both legs over NATS when both workers registered for it. KV
        # transfer stays engine<->engine (bootstrap_room in the bodies), so the
        # delivery channel doesn't matter; just publish each body and stream D.
        if (
            self.nats_client is not None
            and p.request_transport == "nats"
            and d.request_transport == "nats"
        ):
            # Optional admission throttle (INFERA_NATS_REQ_MAX_PENDING): refuse
            # if either leg's worker is at its in-NATS backlog limit.
            if not (
                await self.nats_client.admit(p.worker_id)
                and await self.nats_client.admit(d.worker_id)
            ):
                self.policy.on_request_finished(p_target.route_key, p_blocks)
                self.policy.on_request_finished(d_target.route_key, d_blocks)
                obs["outcome"] = "429"
                return JSONResponse(
                    content={"error": "PD worker request backlog over limit"},
                    status_code=429,
                    headers={"Retry-After": "1"},
                )
            return await self._concurrent_nats(
                obs,
                p_target,
                p_blocks,
                p_url,
                d_target,
                d_blocks,
                d_url,
                p_body,
                d_body,
                stream,
                p_headers,
                d_headers,
            )

        if stream:
            obs["outcome"] = "ok"  # commit at hand-off
            return StreamingResponse(
                self._stream_dual(
                    p_target,
                    p_blocks,
                    d_target,
                    d_blocks,
                    p_url,
                    d_url,
                    p_body,
                    d_body,
                    p_headers,
                    d_headers,
                ),
                media_type="text/event-stream",
            )

        try:

            async def _post(url, leg, worker_id, leg_body, leg_headers):
                with metrics.track_pd_leg(leg=leg, worker_id=worker_id):
                    return await self._client.post(url, json=leg_body, headers=leg_headers)

            try:
                p_resp, d_resp = await asyncio.gather(
                    _post(p_url, "prefill", p.worker_id, p_body, p_headers),
                    _post(d_url, "decode", d.worker_id, d_body, d_headers),
                )
            except httpx.HTTPError as exc:
                obs["outcome"] = "502"
                metrics.pd_bootstrap_failures_total.labels(reason="worker_unreachable").inc()
                return JSONResponse(
                    content={"error": f"PD request failed: {exc}"},
                    status_code=502,
                )

            if p_resp.status_code >= 400:
                logger.warning(
                    "prefill worker %s returned %d (decode may fail)",
                    p.worker_id,
                    p_resp.status_code,
                )
                metrics.pd_bootstrap_failures_total.labels(reason="prefill_5xx").inc()

            try:
                payload = d_resp.json()
            except ValueError:
                obs["outcome"] = "502"
                return JSONResponse(
                    content={
                        "error": f"decode worker {d.worker_id} returned non-JSON",
                        "raw": d_resp.text[:500],
                    },
                    status_code=502,
                )
            obs["outcome"] = "ok" if d_resp.status_code < 400 else f"{d_resp.status_code // 100}xx"
            return JSONResponse(content=payload, status_code=d_resp.status_code)
        finally:
            self.policy.on_request_finished(p_target.route_key, p_blocks)
            self.policy.on_request_finished(d_target.route_key, d_blocks)

    def _start_prefill_drain_nats(self, p, p_payload):
        """Fire the prefill leg over NATS and drain its reply in the background.
        Must run to completion (never cancel): the prefill engine needs the full
        request to register the bootstrap_room and push KV to decode, exactly
        like the HTTP path. Strong ref guards against GC mid-flight."""

        async def _drain():
            try:
                async for kind, _st, data in self.nats_client.stream(p.worker_id, p_payload):
                    if kind == TYPE_ERROR:
                        logger.warning("prefill leg (nats) %s failed: %s", p.worker_id, data[:200])
                        metrics.pd_bootstrap_failures_total.labels(reason="prefill_exception").inc()
                        return
                    if kind == TYPE_DONE:
                        return
            except Exception as exc:
                logger.warning("prefill nats drain %s failed: %s", p.worker_id, exc)

        task = asyncio.create_task(_drain(), name="nats-prefill-drain")
        self._pending_prefill_tasks.add(task)
        task.add_done_callback(self._pending_prefill_tasks.discard)
        return task

    async def _concurrent_nats(
        self,
        obs,
        p_target,
        p_blocks,
        p_url,
        d_target,
        d_blocks,
        d_url,
        p_body,
        d_body,
        stream,
        p_headers,
        d_headers,
    ) -> Response:
        """Concurrent PD over NATS: publish p_body to prefill + d_body to decode
        on their per-instance subjects, stream decode back. KV transfer is
        engine<->engine (mori) via the bootstrap_room in the bodies."""
        p, d = p_target.worker, d_target.worker
        path = p_url[len(p.url) :] or "/v1/chat/completions"
        p_payload = {"path": path, "stream": False, "headers": p_headers, "body": p_body}
        d_payload = {"path": path, "stream": stream, "headers": d_headers, "body": d_body}
        p_task = self._start_prefill_drain_nats(p, p_payload)

        if stream:
            obs["outcome"] = "ok"
            return StreamingResponse(
                self._stream_dual_nats(p_target, p_blocks, d_target, d_blocks, d_payload, p_task),
                media_type="text/event-stream",
            )

        try:
            chunks: list[bytes] = []
            status = 200
            async for kind, st, data in self.nats_client.stream(d.worker_id, d_payload):
                if kind == TYPE_DATA:
                    chunks.append(data)
                elif kind == TYPE_ERROR:
                    # st carries 504 on inactivity timeout; worker errors -> 502.
                    code = st or 502
                    obs["outcome"] = str(code)
                    return JSONResponse(
                        content={
                            "error": f"decode {d.worker_id} nats failed",
                            "raw": data[:500].decode("utf-8", "replace"),
                        },
                        status_code=code,
                    )
                else:  # done
                    status = st or 200
                    break
            raw = b"".join(chunks)
            try:
                payload = json.loads(raw) if raw else {}
            except ValueError:
                obs["outcome"] = "502"
                return JSONResponse(
                    content={
                        "error": f"decode {d.worker_id} non-JSON over nats",
                        "raw": raw[:500].decode("utf-8", "replace"),
                    },
                    status_code=502,
                )
            obs["outcome"] = "ok" if status < 400 else f"{status // 100}xx"
            return JSONResponse(content=payload, status_code=status)
        finally:
            try:
                await asyncio.shield(p_task)
            except (asyncio.CancelledError, Exception):
                pass
            self.policy.on_request_finished(p_target.route_key, p_blocks)
            self.policy.on_request_finished(d_target.route_key, d_blocks)

    async def _stream_dual_nats(self, p_target, p_blocks, d_target, d_blocks, d_payload, p_task):
        """Stream decode's reply over NATS while prefill drains in background."""
        d = d_target.worker
        try:
            async for kind, _st, data in self.nats_client.stream(d.worker_id, d_payload):
                if kind == TYPE_DATA:
                    if data:
                        yield data
                elif kind == TYPE_ERROR:
                    logger.warning("decode (nats) %s stream failed: %s", d.worker_id, data[:200])
                    metrics.pd_bootstrap_failures_total.labels(reason="decode_stream_broken").inc()
                    yield (
                        f'data: {{"error":"decode {d.worker_id} nats stream failed"}}\n\n'
                    ).encode()
                    return
                else:  # done
                    return
        finally:
            try:
                await asyncio.shield(p_task)
            except asyncio.CancelledError:
                logger.debug("prefill nats task cancelled (parent torn down)")
            except Exception:
                pass
            self.policy.on_request_finished(p_target.route_key, p_blocks)
            self.policy.on_request_finished(d_target.route_key, d_blocks)

    async def _dispatch_serial(
        self,
        obs,
        proto,
        base,
        hints,
        p_target,
        p_blocks,
        p_url,
        d_target,
        d_blocks,
        d_url,
        room_id,
        stream,
        forged_id: str | None,
    ) -> Response:
        """Serial-pull topology: D needs handoff fields from P's response
        before its body can be assembled, so the two legs cannot be
        parallelised. Flow:

          1. ``annotate_prefill`` → POST P, await full JSON response.
          2. ``extract_handoff(P's body)`` → connector-specific dict
             (e.g. ``remote_block_ids``, ``remote_engine_id`` for MoRIIO).
          3. ``annotate_decode(handoff)`` → POST D, stream its response.

        On P 4xx/5xx we never call D — its body would be ill-formed and
        we'd just burn engine queue slots. ``track_pd_leg`` still wraps
        each leg so timings line up with the concurrent path.
        """
        p, d = p_target.worker, d_target.worker
        p_headers = _leg_headers(forged_id, p_target)
        d_headers = _leg_headers(forged_id, d_target)
        # P-leg first. Any failure here finishes BOTH workers (we never
        # call D, so D's slot is freed immediately). Once P succeeds we
        # finish P right away — serial-pull means it's truly done at
        # that point, no background task to drain.
        try:
            p_body = inject_engine_priority(
                proto.annotate_prefill(base, p, d, room_id), hints, p.engine
            )
        except ValueError as exc:
            self.policy.on_request_finished(p_target.route_key, p_blocks)
            self.policy.on_request_finished(d_target.route_key, d_blocks)
            obs["outcome"] = "500"
            metrics.pd_bootstrap_failures_total.labels(reason="protocol_annotate_failed").inc()
            return JSONResponse(content={"error": str(exc)}, status_code=500)

        p_failed = False
        try:
            with metrics.track_pd_leg(leg="prefill", worker_id=p.worker_id):
                try:
                    p_resp = await self._client.post(p_url, json=p_body, headers=p_headers)
                except httpx.HTTPError as exc:
                    p_failed = True
                    obs["outcome"] = "502"
                    metrics.pd_bootstrap_failures_total.labels(reason="prefill_unreachable").inc()
                    return JSONResponse(
                        content={"error": f"prefill leg failed: {exc}"},
                        status_code=502,
                    )

            if p_resp.status_code >= 400:
                p_failed = True
                obs["outcome"] = f"{p_resp.status_code // 100}xx"
                metrics.pd_bootstrap_failures_total.labels(
                    reason=f"prefill_{p_resp.status_code // 100}xx"
                ).inc()
                logger.warning(
                    "prefill worker %s returned %d in serial-pull; aborting",
                    p.worker_id,
                    p_resp.status_code,
                )
                return JSONResponse(
                    content={"error": (f"prefill {p_resp.status_code}: {p_resp.text[:500]}")},
                    status_code=p_resp.status_code,
                )

            try:
                p_payload = p_resp.json()
            except ValueError:
                p_failed = True
                obs["outcome"] = "502"
                return JSONResponse(
                    content={
                        "error": f"prefill worker {p.worker_id} returned non-JSON",
                        "raw": p_resp.text[:500],
                    },
                    status_code=502,
                )

            try:
                handoff = proto.extract_handoff(p_payload)
            except (KeyError, ValueError) as exc:
                p_failed = True
                obs["outcome"] = "502"
                metrics.pd_bootstrap_failures_total.labels(reason="handoff_extract_failed").inc()
                return JSONResponse(
                    content={
                        "error": f"handoff extraction failed: {exc}",
                        "prefill_payload": p_payload,
                    },
                    status_code=502,
                )
        finally:
            if p_failed:
                self.policy.on_request_finished(p_target.route_key, p_blocks)
                self.policy.on_request_finished(d_target.route_key, d_blocks)

        # P done & freed. From here on only D is in-flight.
        self.policy.on_request_finished(p_target.route_key, p_blocks)

        try:
            d_body = inject_disagg_prefill_dp_rank(
                inject_engine_priority(
                    proto.annotate_decode(base, p, d, room_id, handoff),
                    hints,
                    d.engine,
                ),
                prefill_target=p_target,
                decode_engine=d.engine,
            )
        except ValueError as exc:
            self.policy.on_request_finished(d_target.route_key, d_blocks)
            obs["outcome"] = "500"
            metrics.pd_bootstrap_failures_total.labels(reason="protocol_annotate_failed").inc()
            return JSONResponse(content={"error": str(exc)}, status_code=500)

        if stream:
            obs["outcome"] = "ok"  # commit at hand-off
            # No prefill task to babysit (already finished); D's stream
            # is self-contained. _stream_decode_only's finally finishes D.
            return StreamingResponse(
                self._stream_decode_only(d_target, d_blocks, d_url, d_body, d_headers),
                media_type="text/event-stream",
            )

        try:
            with metrics.track_pd_leg(leg="decode", worker_id=d.worker_id):
                try:
                    d_resp = await self._client.post(d_url, json=d_body, headers=d_headers)
                except httpx.HTTPError as exc:
                    obs["outcome"] = "502"
                    metrics.pd_bootstrap_failures_total.labels(reason="decode_unreachable").inc()
                    return JSONResponse(
                        content={"error": f"decode leg failed: {exc}"},
                        status_code=502,
                    )

            try:
                d_payload = d_resp.json()
            except ValueError:
                obs["outcome"] = "502"
                return JSONResponse(
                    content={
                        "error": f"decode worker {d.worker_id} returned non-JSON",
                        "raw": d_resp.text[:500],
                    },
                    status_code=502,
                )
            obs["outcome"] = "ok" if d_resp.status_code < 400 else f"{d_resp.status_code // 100}xx"
            return JSONResponse(content=d_payload, status_code=d_resp.status_code)
        finally:
            self.policy.on_request_finished(d_target.route_key, d_blocks)

    async def _stream_decode_only(
        self,
        d_target,
        d_blocks: list[int],
        d_url: str,
        d_body: dict,
        d_headers: dict[str, str] | None = None,
    ):
        """Serial-pull streaming path — P has already finished and freed
        its scheduler slot, so all we need to do is stream D and not
        babysit a background prefill task. Reuses the same pre-flight
        retry + post-[DONE] suppression as `_stream_dual` for parity.
        """
        d_resp: httpx.Response | None = None
        done_seen = False
        try:
            try:
                d_resp = await self._open_decode_stream(d_url, d_body, d_headers)
            except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                logger.warning(
                    "decode leg %s unreachable after %d retries: %s: %s",
                    d_url,
                    self._DECODE_OPEN_MAX_RETRIES,
                    type(exc).__name__,
                    exc or "<no message>",
                )
                metrics.pd_bootstrap_failures_total.labels(reason="decode_unreachable").inc()
                err = json.dumps({"error": f"decode unreachable: {type(exc).__name__}: {exc}"})
                yield f"data: {err}\n\n".encode()
                return

            if d_resp.status_code >= 400:
                try:
                    body_bytes = await d_resp.aread()
                except Exception:
                    body_bytes = b""
                logger.warning(
                    "decode leg %s returned %d before streaming: %r",
                    d_url,
                    d_resp.status_code,
                    body_bytes[:500],
                )
                metrics.pd_bootstrap_failures_total.labels(
                    reason=f"decode_{d_resp.status_code // 100}xx"
                ).inc()
                err = json.dumps(
                    {
                        "error": (
                            f"decode {d_resp.status_code}: "
                            f"{body_bytes.decode('utf-8', errors='replace')[:500]}"
                        )
                    }
                )
                yield f"data: {err}\n\n".encode()
                return

            _DONE_NEEDLE = b"data: [DONE]"
            _TAIL_KEEP = len(_DONE_NEEDLE) - 1
            tail = b""
            try:
                async for chunk in d_resp.aiter_raw():
                    if not done_seen:
                        window = tail + chunk
                        if _DONE_NEEDLE in window:
                            done_seen = True
                        tail = window[-_TAIL_KEEP:]
                    yield chunk
            except httpx.HTTPError as exc:
                if done_seen:
                    logger.debug(
                        "decode stream from %s closed after [DONE] (%s)",
                        d_url,
                        type(exc).__name__,
                    )
                    return
                logger.warning(
                    "decode stream from %s failed mid-response: %s: %s",
                    d_url,
                    type(exc).__name__,
                    exc or "<no message>",
                )
                metrics.pd_bootstrap_failures_total.labels(reason="decode_stream_broken").inc()
                err = json.dumps({"error": f"decode stream failed: {type(exc).__name__}: {exc}"})
                yield f"data: {err}\n\n".encode()
        finally:
            if d_resp is not None:
                try:
                    await d_resp.aclose()
                except Exception:
                    pass
            self.policy.on_request_finished(d_target.route_key, d_blocks)

    async def _open_decode_stream(
        self,
        d_url: str,
        d_body: dict,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """POST the decode leg, retrying on pre-flight transport errors.

        Returns the streaming Response; caller must aclose() it exactly
        once. Pre-flight errors (ConnectError / ReadError on headers /
        WriteError mid-body / RemoteProtocolError) mean the engine has
        NOT begun processing the request, so re-sending the same body
        with the same bootstrap_room is idempotent.
        """
        backoff = self._DECODE_OPEN_INITIAL_BACKOFF_S
        last_exc: BaseException | None = None
        for attempt in range(self._DECODE_OPEN_MAX_RETRIES + 1):
            req = self._client.build_request("POST", d_url, json=d_body, headers=headers)
            try:
                resp = await self._client.send(req, stream=True)
            except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt >= self._DECODE_OPEN_MAX_RETRIES:
                    raise
                logger.info(
                    "decode leg open retry %d/%d for %s: %s: %s",
                    attempt + 1,
                    self._DECODE_OPEN_MAX_RETRIES,
                    d_url,
                    type(exc).__name__,
                    exc or "<no message>",
                )
                metrics.pd_bootstrap_failures_total.labels(reason="decode_open_retried").inc()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._DECODE_OPEN_MAX_BACKOFF_S)
                continue
            return resp
        # Loop always returns or raises; this satisfies type checkers.
        assert last_exc is not None
        raise last_exc

    async def _stream_dual(
        self,
        p_target,
        p_blocks: list[int],
        d_target,
        d_blocks: list[int],
        p_url: str,
        d_url: str,
        p_body: dict,
        d_body: dict,
        p_headers: dict[str, str] | None = None,
        d_headers: dict[str, str] | None = None,
    ):
        """Stream D's response while P runs concurrently in the background.

        P's HTTP connection stays open as long as KV is being transferred;
        we don't read its body, only its task lifetime matters. p_body and
        d_body differ only in engine-specific priority injection.
        """
        p = p_target.worker
        # Never cancel p_task: closing the body drops the bootstrap_room
        # handoff → decode stuck on KVPoll 300s. Strong ref + shield guard
        # against GC and parent cancellation.
        p_task = asyncio.create_task(self._client.post(p_url, json=p_body, headers=p_headers))
        self._pending_prefill_tasks.add(p_task)
        p_task.add_done_callback(self._pending_prefill_tasks.discard)
        # Once we've forwarded "data: [DONE]" downstream, any subsequent
        # httpx.ReadError is the client closing its half of a successful
        # response — drop silently instead of warning.
        done_seen = False
        d_resp: httpx.Response | None = None
        try:
            try:
                # Pre-flight is retryable (engine hasn't seen us); a
                # mid-stream read is not (would double-bill).
                try:
                    d_resp = await self._open_decode_stream(d_url, d_body, d_headers)
                except (httpx.TransportError, httpx.RemoteProtocolError) as exc:
                    # Engine never saw this body; finally still drains p_task.
                    # Emit a clean SSE error so the client records Failed.
                    logger.warning(
                        "decode leg %s unreachable after %d retries: %s: %s",
                        d_url,
                        self._DECODE_OPEN_MAX_RETRIES,
                        type(exc).__name__,
                        exc or "<no message>",
                    )
                    metrics.pd_bootstrap_failures_total.labels(reason="decode_unreachable").inc()
                    # json.dumps: exc text may contain chars that break SSE.
                    err = json.dumps({"error": f"decode unreachable: {type(exc).__name__}: {exc}"})
                    yield f"data: {err}\n\n".encode()
                    return

                if d_resp.status_code >= 400:
                    # Engine accepted but rejected; surface its body verbatim.
                    try:
                        body_bytes = await d_resp.aread()
                    except Exception:
                        body_bytes = b""
                    logger.warning(
                        "decode leg %s returned %d before streaming: %r",
                        d_url,
                        d_resp.status_code,
                        body_bytes[:500],
                    )
                    metrics.pd_bootstrap_failures_total.labels(
                        reason=f"decode_{d_resp.status_code // 100}xx"
                    ).inc()
                    err = json.dumps(
                        {
                            "error": (
                                f"decode {d_resp.status_code}: "
                                f"{body_bytes.decode('utf-8', errors='replace')[:500]}"
                            )
                        }
                    )
                    yield f"data: {err}\n\n".encode()
                    return

                # aiter_raw yields raw bytes, so "data: [DONE]" can straddle
                # chunks; a tail buffer keeps the match across splits.
                _DONE_NEEDLE = b"data: [DONE]"
                _TAIL_KEEP = len(_DONE_NEEDLE) - 1
                tail = b""
                async for chunk in d_resp.aiter_raw():
                    if not done_seen:
                        window = tail + chunk
                        if _DONE_NEEDLE in window:
                            done_seen = True
                        tail = window[-_TAIL_KEEP:]
                    yield chunk
            except httpx.HTTPError as exc:
                if done_seen:
                    # Engine has already sent [DONE]; this is the client
                    # tearing down a successful response. Drop silently.
                    logger.debug(
                        "decode stream from %s closed after [DONE] (%s)",
                        d_url,
                        type(exc).__name__,
                    )
                    return
                # Mid-stream failure: can't retry (would double-bill). Emit an
                # SSE error so truncation isn't mistaken for success; include
                # exc class since httpx stream errors often stringify to "".
                logger.warning(
                    "decode stream from %s failed mid-response: %s: %s",
                    d_url,
                    type(exc).__name__,
                    exc or "<no message>",
                )
                metrics.pd_bootstrap_failures_total.labels(reason="decode_stream_broken").inc()
                err = json.dumps({"error": f"decode stream failed: {type(exc).__name__}: {exc}"})
                yield f"data: {err}\n\n".encode()
        finally:
            if d_resp is not None:
                try:
                    await d_resp.aclose()
                except Exception:
                    pass
            # Await p_task (never cancel; shield from parent cancel). Log
            # outcomes since a silent prefill drop costs 300s: parent
            # cancel→DEBUG, raise/4xx-5xx→WARN.
            try:
                p_resp = await asyncio.shield(p_task)
            except asyncio.CancelledError:
                logger.debug("prefill task cancelled (parent torn down)")
            except Exception as exc:
                logger.warning(
                    "prefill leg %s for %s failed: %s: %s",
                    p.worker_id,
                    p_url,
                    type(exc).__name__,
                    exc or "<no message>",
                )
                metrics.pd_bootstrap_failures_total.labels(reason="prefill_exception").inc()
            else:
                if getattr(p_resp, "status_code", 0) >= 400:
                    logger.warning(
                        "prefill leg %s returned %d (decode will hang on KVPoll)",
                        p.worker_id,
                        p_resp.status_code,
                    )
                    metrics.pd_bootstrap_failures_total.labels(reason="prefill_5xx").inc()
            self.policy.on_request_finished(p_target.route_key, p_blocks)
            self.policy.on_request_finished(d_target.route_key, d_blocks)
