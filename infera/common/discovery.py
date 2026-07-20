###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Etcd-backed worker discovery (server side), via the etcd v3 HTTP/JSON gateway.

Talks to ``http://<endpoint>/v3/...`` with httpx -- no ``etcd3`` / protobuf /
grpc dependency. Etcd serves the gateway on the same client port (2379 by
default), so this is the same endpoint a normal etcd client would hit.

Workers ``PUT`` their own ``WorkerInfo`` to ``<prefix>/<id>`` with a lease
(see :class:`infera.common.registration.RegistrationClient`); the server
snapshots the prefix on start and then opens a long-lived ``Watch`` stream
to mirror live PUT/DELETE events into a local :class:`WorkerPool`. The
router consumes that pool unchanged.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable

import httpx

from infera.common.worker_pool import (
    CanaryMismatch,
    CanaryVerifier,
    DisaggMode,
    EngineType,
    KvRegistrationMetadata,
    WorkerInfo,
    WorkerPool,
    WorkerStatus,
)

logger = logging.getLogger(__name__)

DEFAULT_PREFIX = "/infera/workers/"


def worker_info_from_json(data: dict) -> WorkerInfo:
    """Build a WorkerInfo from a decoded registration payload.

    Shared by every discovery backend (etcd Registry, KubernetesRegistry) so
    the on-the-wire worker record is identical regardless of transport. Raises
    on malformed input; callers log + skip.
    """
    kv_raw = data.get("kv")
    kv = KvRegistrationMetadata.from_dict(kv_raw) if kv_raw else None
    return WorkerInfo(
        worker_id=data["worker_id"],
        url=data["url"],
        model_name=data["model_name"],
        engine=EngineType(data.get("engine", EngineType.SGLANG.value)),
        status=WorkerStatus(data.get("status", WorkerStatus.ACTIVE.value)),
        disagg_mode=DisaggMode(data.get("disagg_mode", DisaggMode.MIXED.value)),
        disagg_meta=dict(data.get("disagg_meta") or {}),
        kv_events_endpoint=data.get("kv_events_endpoint"),
        kv_block_size=data.get("kv_block_size"),
        dp_rank=data.get("dp_rank"),
        dp_size=data.get("dp_size"),
        request_transport=data.get("request_transport", "http"),
        kv=kv,
    )


def _normalize_endpoint(endpoint: str) -> str:
    """Accept ``host:port`` / ``host`` / ``http(s)://...`` and return a base URL."""
    if endpoint.startswith(("http://", "https://")):
        return endpoint.rstrip("/")
    if ":" not in endpoint:
        endpoint = f"{endpoint}:2379"
    return f"http://{endpoint}"


def _b64(s: str) -> str:
    return base64.b64encode(s.encode()).decode()


def _b64bytes(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _unb64(s: str) -> bytes:
    return base64.b64decode(s)


def _range_end_for_prefix(prefix: str) -> bytes:
    """etcd convention: range_end = prefix with last byte +1 selects the whole prefix."""
    pb = prefix.encode()
    if not pb:
        return b"\x00"
    return pb[:-1] + bytes([pb[-1] + 1])


class Registry:
    """Server-side worker registry backed by an etcd cluster.

    Pass ``on_worker_added`` / ``on_worker_removed`` to react to fleet
    changes (e.g. start/stop a KvEventClient subscription, hook in a
    snapshot reconciler). Both callbacks are invoked synchronously
    inside the watch loop, so keep them cheap or fan out to a task.
    Exceptions are logged + swallowed so one bad listener doesn't
    break the watch loop.

    Pass ``canary_verifier`` to enable cross-worker tokenizer canary
    verification for the nested ``kv`` registration block (see
    `KvRegistrationMetadata`). Workers whose canary mismatches the
    first-registered worker for the same model are rejected.
    """

    def __init__(
        self,
        endpoint: str,
        prefix: str = DEFAULT_PREFIX,
        on_worker_added: Callable[[WorkerInfo], None] | None = None,
        on_worker_removed: Callable[[str], None] | None = None,
        *,
        canary_verifier: CanaryVerifier | None = None,
    ) -> None:
        self._base = _normalize_endpoint(endpoint)
        self._prefix = prefix if prefix.endswith("/") else prefix + "/"
        self._range_end = _range_end_for_prefix(self._prefix)
        self._pool = WorkerPool()
        self._on_added = on_worker_added
        self._on_removed = on_worker_removed
        self._canary = canary_verifier if canary_verifier is not None else CanaryVerifier()
        self._http: httpx.AsyncClient | None = None
        self._watch_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        # Counters for /v1/kv-stats and operator alerts.
        self.canary_rejections = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._http = httpx.AsyncClient(base_url=self._base, timeout=10.0)
        self._stop_event = asyncio.Event()

        # 1. snapshot
        resp = await self._http.post(
            "/v3/kv/range",
            json={"key": _b64(self._prefix), "range_end": _b64bytes(self._range_end)},
        )
        resp.raise_for_status()
        kvs = resp.json().get("kvs", []) or []
        for kv in kvs:
            key = _unb64(kv["key"]).decode()
            value = _unb64(kv["value"])
            self._upsert(key, value)
        logger.info("etcd snapshot: %d worker(s) under %s", len(self._pool), self._prefix)

        # 2. watch (background task, auto-reconnect)
        self._watch_task = asyncio.create_task(self._watch_loop(), name="etcd-watch")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._watch_task is not None:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except (asyncio.CancelledError, Exception):
                pass
            self._watch_task = None
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def pool(self) -> WorkerPool:
        return self._pool

    def list_all(self) -> list[WorkerInfo]:
        return self._pool.list_all()

    # ------------------------------------------------------------------
    # Watch loop
    # ------------------------------------------------------------------

    async def _watch_loop(self) -> None:
        """Open a streaming Watch, mirror events. Reconnect on failure."""
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._watch_once()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("etcd watch disconnected: %s; retry in %.1fs", exc, backoff)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)

    async def _watch_once(self) -> None:
        # etcd v3 Watch is a bidi gRPC stream; the JSON gateway accepts a
        # newline-delimited JSON request body.
        create_req = {
            "create_request": {
                "key": _b64(self._prefix),
                "range_end": _b64bytes(self._range_end),
            }
        }
        # Use a fresh client with no read timeout for the long-lived stream.
        async with httpx.AsyncClient(base_url=self._base, timeout=None) as client:
            async with client.stream(
                "POST",
                "/v3/watch",
                content=(json.dumps(create_req) + "\n").encode(),
                headers={"Content-Type": "application/json"},
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    self._dispatch_watch(msg)
                    if self._stop_event.is_set():
                        return

    def _dispatch_watch(self, msg: dict) -> None:
        # The gateway wraps each gRPC message under a top-level "result" key.
        result = msg.get("result", msg)
        events = result.get("events") or []
        for ev in events:
            ev_type = ev.get("type", "PUT")  # gateway omits "type" for PUT
            kv = ev.get("kv", {}) or {}
            key_b64 = kv.get("key")
            if not key_b64:
                continue
            key = _unb64(key_b64).decode()
            if ev_type == "DELETE":
                worker_id = key[len(self._prefix) :]
                existing = self._pool.get(worker_id)
                if existing is not None:
                    self._pool.remove(worker_id)
                    # If no other workers remain for this model, forget the
                    # canary so a fresh registration can re-set it. This lets
                    # operators recover from a misconfiguration by removing
                    # all workers and restarting.
                    remaining = [
                        w for w in self._pool.list_all() if w.model_name == existing.model_name
                    ]
                    if not remaining:
                        self._canary.forget(existing.model_name)
                    logger.info("etcd: worker %s removed (lease expired or revoked)", worker_id)
                    if self._on_removed is not None:
                        try:
                            self._on_removed(worker_id)
                        except Exception:
                            logger.exception("on_worker_removed callback failed")
            else:  # PUT
                value = _unb64(kv.get("value", "")) if kv.get("value") else b""
                self._upsert(key, value)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _upsert(self, key: str, raw_value: bytes | str) -> None:
        worker_id = key[len(self._prefix) :]
        try:
            data = json.loads(raw_value)
            info = worker_info_from_json(data)
        except Exception as exc:
            logger.warning("etcd: bad value at %s: %s", key, exc)
            return

        # Cross-worker tokenizer canary verification (see 03-data-model.md
        # § "Cross-worker tokenizer consistency"). Skipped for workers
        # that registered without a kv block — they're pre-Phase-1 or
        # opted out of prefix-cache participation.
        if info.kv is not None and info.kv.tokenizer_canary:
            try:
                self._canary.verify(
                    model_name=info.model_name,
                    worker_id=info.worker_id,
                    canary=info.kv.tokenizer_canary,
                )
            except CanaryMismatch as exc:
                self.canary_rejections += 1
                logger.error(
                    "etcd: REJECTING worker %s — %s. Registration ignored; "
                    "operator must align tokenizers across workers serving %r.",
                    info.worker_id,
                    exc,
                    info.model_name,
                )
                return

        existed = self._pool.get(worker_id) is not None
        self._pool.add(info)
        logger.info(
            "etcd: worker %s %s (%s, model=%s, disagg=%s, kv_events=%s, kv=%s)",
            worker_id,
            "updated" if existed else "registered",
            info.url,
            info.model_name,
            info.disagg_mode,
            info.kv_events_endpoint,
            "yes" if info.kv is not None else "no",
        )
        # NOTE: on_added only fires on FIRST registration of a worker_id.
        # If a worker is re-registered with a changed endpoint (e.g. its
        # dynamically allocated kv_events_endpoint port shifts after a fast
        # restart), subscribers won't auto-reconnect to the new endpoint.
        # In practice the etcd lease (~30s TTL) expires before any restart
        # completes, triggering a DELETE -> on_removed first, so this is
        # only a concern for sub-second restart cycles.
        if not existed and self._on_added is not None:
            try:
                self._on_added(info)
            except Exception:
                logger.exception("on_worker_added callback failed")
