###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import asyncio
import json
import logging

import httpx

from infera.common.discovery import DEFAULT_PREFIX, _b64, _normalize_endpoint
from infera.common.worker_pool import WorkerStatus
from infera.engine.base import EngineConfig

logger = logging.getLogger(__name__)

_DEFAULT_LEASE_TTL = 30  # seconds


def build_worker_payload(config: EngineConfig, *, status: WorkerStatus | None = None) -> dict:
    """Build the worker registration record shared by every backend.

    The same dict is PUT to etcd (RegistrationClient) or stored in the worker
    Pod annotation (K8sRegistrationClient), so the server-side parse
    (discovery.worker_info_from_json) is transport-agnostic.

    ``status`` is omitted for a healthy worker, which keeps the record identical
    to what older workers wrote and lets the parser's ACTIVE default stand. It
    is set only to announce DRAINING, so the field's presence means something
    happened rather than being ambient.
    """
    worker_id = f"{config.host}:{config.port}"
    payload: dict = {
        "worker_id": worker_id,
        "url": f"http://{config.host}:{config.port}",
        "model_name": config.model_name,
        "engine": config.engine,
        "disagg_mode": config.disagg_mode,
        "disagg_meta": config.disagg_meta,
        "kv_events_endpoint": config.kv_events_endpoint,
        "kv_block_size": config.kv_block_size,
        "dp_rank": config.dp_rank,
        "dp_size": config.dp_size,
        "request_transport": getattr(config, "request_transport", "http"),
    }
    if status is not None and status is not WorkerStatus.ACTIVE:
        payload["status"] = status.value
    if config.kv is not None:
        payload["kv"] = config.kv.to_dict()
    return payload


class RegistrationClient:
    """Worker-side self-registration via an etcd lease (HTTP/JSON gateway)."""

    #: etcd knows only that a record exists and its lease is unexpired; nothing
    #: outside the worker observes that the process is going away. So the
    #: worker has to say so itself, or there is no way to express "still
    #: finishing what I have, send me nothing new" -- only present or absent.
    announces_draining = True

    def __init__(
        self,
        endpoint: str,
        prefix: str = DEFAULT_PREFIX,
        lease_ttl: int = _DEFAULT_LEASE_TTL,
    ) -> None:
        self._base = _normalize_endpoint(endpoint)
        self._prefix = prefix if prefix.endswith("/") else prefix + "/"
        self._lease_ttl = lease_ttl
        self._http = httpx.AsyncClient(base_url=self._base, timeout=10.0)
        self._lease_id: int | None = None
        self._worker_id: str | None = None
        self._key: str | None = None
        self._config: EngineConfig | None = None  # cached for re-register on lease loss

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register(self, config: EngineConfig) -> str:
        worker_id = f"{config.host}:{config.port}"
        key = self._prefix + worker_id
        value = json.dumps(build_worker_payload(config))

        # 1. grant lease
        r = await self._http.post("/v3/lease/grant", json={"TTL": self._lease_ttl})
        r.raise_for_status()
        lease_id = int(r.json()["ID"])

        # 2. put with lease
        r = await self._http.post(
            "/v3/kv/put",
            json={"key": _b64(key), "value": _b64(value), "lease": lease_id},
        )
        r.raise_for_status()

        self._lease_id = lease_id
        self._worker_id = worker_id
        self._key = key
        self._config = config
        logger.info(
            "registered with etcd as worker %s (lease=%d, ttl=%ds, engine=%s, disagg=%s)",
            worker_id,
            lease_id,
            self._lease_ttl,
            config.engine,
            config.disagg_mode,
        )
        return worker_id

    async def announce_draining(self) -> bool:
        """Rewrite the record as DRAINING, keeping the lease alive.

        ``list_active`` filters DRAINING out, so this stops new work being
        routed here without deleting the record — which is the difference that
        matters during a shutdown. A worker that simply vanishes is
        indistinguishable from one that crashed; one that is visibly draining
        tells an operator (and ``/v1/workers``) that a rolling update is
        proceeding normally and roughly how far along it is.

        On the Kubernetes backend this is largely redundant with the registry's
        ``deletionTimestamp`` check, which removes a condemned Pod without the
        worker having to say anything. It is not redundant on etcd, where
        nothing else observes that the process is going away.
        """
        if self._lease_id is None or self._key is None or self._config is None:
            return False
        try:
            value = json.dumps(build_worker_payload(self._config, status=WorkerStatus.DRAINING))
            r = await self._http.post(
                "/v3/kv/put",
                json={"key": _b64(self._key), "value": _b64(value), "lease": self._lease_id},
            )
            r.raise_for_status()
            logger.info("worker %s announced DRAINING", self._worker_id)
            return True
        except Exception as exc:  # noqa: BLE001 - shutdown must continue
            logger.warning("could not announce DRAINING: %s", exc)
            return False

    async def deregister(self) -> None:
        if self._lease_id is not None:
            try:
                await self._http.post("/v3/lease/revoke", json={"ID": self._lease_id})
                logger.info(
                    "deregistered worker %s (lease %d revoked)",
                    self._worker_id,
                    self._lease_id,
                )
            except Exception as exc:
                logger.warning("lease revoke failed: %s", exc)
            self._lease_id = None
        self._worker_id = None
        self._key = None
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def heartbeat_loop(self, interval: float | None = None) -> None:
        """Refresh the etcd lease until cancelled.

        On lease loss (TTL == 0 or HTTP error reporting the lease is gone),
        attempts to re-register using the cached :class:`EngineConfig`.
        """
        interval = interval if interval is not None else max(self._lease_ttl / 3, 1.0)
        while True:
            await asyncio.sleep(interval)
            if self._lease_id is None:
                return
            try:
                # The gateway accepts /v3/lease/keepalive as a streamed request;
                # for a single ping we just send one message and read the
                # synchronous response.
                r = await self._http.post(
                    "/v3/lease/keepalive",
                    json={"ID": self._lease_id},
                )
                if r.status_code != 200:
                    logger.warning("lease keepalive HTTP %d: %s", r.status_code, r.text[:200])
                    continue
                data = r.json().get("result", r.json())
                if int(data.get("TTL", 0)) <= 0:
                    logger.warning(
                        "etcd reports lease %d is gone; re-registering",
                        self._lease_id,
                    )
                    if self._config is not None:
                        try:
                            await self.register(self._config)
                        except Exception as exc:
                            logger.warning("etcd re-register failed: %s", exc)
                    else:
                        return
            except Exception as exc:
                logger.warning("lease keepalive failed: %s", exc)
