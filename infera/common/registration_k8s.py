###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Worker-side self-registration via the Kubernetes API (no etcd).

The ``kubernetes`` backend counterpart of
:class:`infera.common.registration.RegistrationClient`: instead of an etcd
lease + PUT, the worker writes its :class:`build_worker_payload` record into its
own Pod annotation (``infera.amd.com/worker-info``). The server's
:class:`infera.common.discovery_k8s.KubernetesRegistry` watches for it. Pod
lifetime replaces the lease — when the Pod is deleted the record disappears
with it, so there is nothing to revoke.

Requires POD_NAME / POD_NAMESPACE (downward API) and RBAC to patch its own Pod.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from infera.common.discovery_k8s import WORKER_INFO_ANNOTATION
from infera.common.k8s_client import in_cluster_namespace, make_client
from infera.common.registration import build_worker_payload
from infera.engine.base import EngineConfig

logger = logging.getLogger(__name__)

# Re-PATCH cadence: cheap self-heal if something strips the annotation. Pod
# liveness (not this loop) is what deregisters a dead worker.
_DEFAULT_REFRESH = 30.0


class K8sRegistrationClient:
    """Worker-side self-registration by patching its own Pod annotation."""

    def __init__(
        self,
        pod_name: str | None = None,
        namespace: str | None = None,
        refresh_interval: float = _DEFAULT_REFRESH,
    ) -> None:
        self._pod_name = pod_name or os.environ.get("POD_NAME", "")
        self._namespace = namespace or os.environ.get("POD_NAMESPACE") or in_cluster_namespace()
        self._refresh = refresh_interval
        self._http = make_client(timeout=10.0)
        self._worker_id: str | None = None
        self._config: EngineConfig | None = None

    async def _patch_annotation(self, value: str | None) -> None:
        if not self._pod_name:
            raise RuntimeError(
                "K8sRegistrationClient needs POD_NAME (downward API) to patch its own Pod"
            )
        # JSON merge patch; value=None clears the annotation.
        patch = {"metadata": {"annotations": {WORKER_INFO_ANNOTATION: value}}}
        r = await self._http.patch(
            f"/api/v1/namespaces/{self._namespace}/pods/{self._pod_name}",
            content=json.dumps(patch).encode(),
            headers={"Content-Type": "application/merge-patch+json"},
        )
        r.raise_for_status()

    async def register(self, config: EngineConfig) -> str:
        worker_id = f"{config.host}:{config.port}"
        await self._patch_annotation(json.dumps(build_worker_payload(config)))
        self._worker_id = worker_id
        self._config = config
        logger.info(
            "registered via k8s Pod annotation: pod=%s/%s worker=%s (engine=%s, disagg=%s)",
            self._namespace,
            self._pod_name,
            worker_id,
            config.engine,
            config.disagg_mode,
        )
        return worker_id

    async def deregister(self) -> None:
        # Best-effort: clear the annotation so a terminating-but-lingering Pod
        # stops being routed before its DELETE event lands.
        try:
            await self._patch_annotation(None)
            logger.info("deregistered worker %s (annotation cleared)", self._worker_id)
        except Exception as exc:
            logger.warning("k8s deregister failed (pod likely terminating): %s", exc)
        self._worker_id = None
        try:
            await self._http.aclose()
        except Exception:
            pass

    async def heartbeat_loop(self, interval: float | None = None) -> None:
        """Periodically re-assert the annotation (self-heal); never expires it."""
        interval = interval if interval is not None else self._refresh
        while True:
            await asyncio.sleep(interval)
            if self._config is None:
                return
            try:
                await self._patch_annotation(json.dumps(build_worker_payload(self._config)))
            except Exception as exc:
                logger.warning("k8s annotation refresh failed: %s", exc)
