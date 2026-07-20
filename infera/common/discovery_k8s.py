###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Kubernetes-native worker discovery (server side).

The ``kubernetes`` discovery backend — the infera analogue of dynamo's
``discoveryBackend: kubernetes``. Instead of an external etcd, workers publish
their :class:`WorkerInfo` into their own Pod annotation
(``infera.amd.com/worker-info``, see :class:`K8sRegistrationClient`), and the
server lists + watches those Pods through the in-cluster API server. The
control-plane etcd is never accessed directly.

Drop-in for :class:`infera.common.discovery.Registry`: same
``start``/``stop``/``pool``/``list_all`` surface and the same
``on_worker_added`` / ``on_worker_removed`` callbacks feeding an unchanged
:class:`WorkerPool`, so the router is oblivious to the backend.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable

import httpx

from infera.common.discovery import worker_info_from_json
from infera.common.k8s_client import in_cluster_namespace, make_client
from infera.common.worker_pool import (
    CanaryMismatch,
    CanaryVerifier,
    WorkerInfo,
    WorkerPool,
)

logger = logging.getLogger(__name__)

# Pod annotation carrying the JSON worker registration record.
WORKER_INFO_ANNOTATION = "infera.amd.com/worker-info"


class KubernetesRegistry:
    """Server-side worker registry backed by the Kubernetes API server.

    Watches Pods matching ``label_selector`` in ``namespace`` and mirrors those
    carrying a ``infera.amd.com/worker-info`` annotation into a WorkerPool.
    Pod deletion (or annotation removal) deregisters the worker — Pod lifetime
    replaces the etcd lease.
    """

    def __init__(
        self,
        label_selector: str,
        namespace: str | None = None,
        on_worker_added: Callable[[WorkerInfo], None] | None = None,
        on_worker_removed: Callable[[str], None] | None = None,
        *,
        canary_verifier: CanaryVerifier | None = None,
    ) -> None:
        self._selector = label_selector
        self._namespace = namespace or in_cluster_namespace()
        self._on_added = on_worker_added
        self._on_removed = on_worker_removed
        self._canary = canary_verifier if canary_verifier is not None else CanaryVerifier()
        self._pool = WorkerPool()
        # pod name -> worker_id, so a DELETE event (keyed by pod) maps back to
        # the worker_id the pool/router track.
        self._pod_to_worker: dict[str, str] = {}
        self._http: httpx.AsyncClient | None = None
        self._watch_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self.canary_rejections = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        self._http = make_client(timeout=10.0)
        self._stop_event = asyncio.Event()
        resource_version = await self._relist()
        # watch (background task, auto-reconnect + re-list on 410/Expired)
        self._watch_task = asyncio.create_task(self._watch_loop(resource_version), name="k8s-watch")

    async def _relist(self) -> str | None:
        """List matching Pods, (re)populate the pool, return the list
        resourceVersion to watch from. Called on start and whenever the watch
        resourceVersion goes stale (HTTP 410 / ERROR event), which etcd
        compaction triggers every few minutes — without this the pool freezes."""
        params = {"labelSelector": self._selector}
        resp = await self._http.get(f"/api/v1/namespaces/{self._namespace}/pods", params=params)
        resp.raise_for_status()
        body = resp.json()
        for pod in body.get("items", []) or []:
            self._handle_pod(pod, deleted=False)
        rv = (body.get("metadata") or {}).get("resourceVersion")
        logger.info(
            "k8s (re)list: %d worker(s) for selector %r in ns %s (rv=%s)",
            len(self._pool),
            self._selector,
            self._namespace,
            rv,
        )
        return rv

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

    async def _watch_loop(self, resource_version: str | None) -> None:
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                # rv None (first run after a stale/410) -> re-list to repopulate
                # the pool + get a fresh resourceVersion before watching.
                if resource_version is None:
                    resource_version = await self._relist()
                resource_version = await self._watch_once(resource_version)
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("k8s watch disconnected: %s; retry in %.1fs", exc, backoff)
                resource_version = None  # force a re-list on the next iteration
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                    return
                except asyncio.TimeoutError:
                    pass
                backoff = min(backoff * 2, 30.0)

    async def _watch_once(self, resource_version: str | None) -> str | None:
        params = {"labelSelector": self._selector, "watch": "true"}
        if resource_version:
            params["resourceVersion"] = resource_version
        # Fresh client with no read timeout for the long-lived stream.
        async with make_client(timeout=None) as client:
            async with client.stream(
                "GET",
                f"/api/v1/namespaces/{self._namespace}/pods",
                params=params,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # A 410/Expired arrives as an ERROR event: the watch rv is
                    # stale (etcd compacted past it). Stop this stream and
                    # return None so _watch_loop re-lists from a fresh rv,
                    # rather than spinning on the same expired rv forever.
                    if event.get("type") == "ERROR":
                        logger.info(
                            "k8s watch stale (re-listing): %s",
                            (event.get("object") or {}).get("message", "ERROR"),
                        )
                        return None
                    rv = self._dispatch_event(event)
                    if rv:
                        resource_version = rv
                    if self._stop_event.is_set():
                        return resource_version
        return resource_version

    def _dispatch_event(self, event: dict) -> str | None:
        ev_type = event.get("type")
        pod = event.get("object") or {}
        rv = ((pod.get("metadata") or {}).get("resourceVersion")) or None
        if ev_type in ("ADDED", "MODIFIED"):
            self._handle_pod(pod, deleted=False)
        elif ev_type == "DELETED":
            self._handle_pod(pod, deleted=True)
        return rv

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _pod_running(pod: dict) -> bool:
        phase = ((pod.get("status") or {}).get("phase")) or ""
        return phase == "Running"

    def _handle_pod(self, pod: dict, *, deleted: bool) -> None:
        meta = pod.get("metadata") or {}
        pod_name = meta.get("name") or ""
        if not pod_name:
            return
        annotations = meta.get("annotations") or {}
        raw = annotations.get(WORKER_INFO_ANNOTATION)

        # Removal: explicit DELETE, pod no longer Running, or annotation gone.
        if deleted or raw is None or not self._pod_running(pod):
            worker_id = self._pod_to_worker.pop(pod_name, None)
            if worker_id is not None:
                self._remove(worker_id)
            return

        try:
            info = worker_info_from_json(json.loads(raw))
        except Exception as exc:
            logger.warning("k8s: bad worker-info annotation on pod %s: %s", pod_name, exc)
            return

        # Cross-worker tokenizer canary verification (same policy as the etcd
        # backend); reject workers whose tokenizer canary mismatches.
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
                    "k8s: REJECTING worker %s — %s. Pod %s ignored; align tokenizers "
                    "across workers serving %r.",
                    info.worker_id,
                    exc,
                    pod_name,
                    info.model_name,
                )
                return

        existed = self._pool.get(info.worker_id) is not None
        self._pod_to_worker[pod_name] = info.worker_id
        self._pool.add(info)
        logger.info(
            "k8s: worker %s %s (pod=%s, %s, model=%s, disagg=%s, kv_events=%s, kv=%s)",
            info.worker_id,
            "updated" if existed else "registered",
            pod_name,
            info.url,
            info.model_name,
            info.disagg_mode,
            info.kv_events_endpoint,
            "yes" if info.kv is not None else "no",
        )
        if not existed and self._on_added is not None:
            try:
                self._on_added(info)
            except Exception:
                logger.exception("on_worker_added callback failed")

    def _remove(self, worker_id: str) -> None:
        existing = self._pool.get(worker_id)
        if existing is None:
            return
        self._pool.remove(worker_id)
        remaining = [w for w in self._pool.list_all() if w.model_name == existing.model_name]
        if not remaining:
            self._canary.forget(existing.model_name)
        logger.info("k8s: worker %s removed (pod deleted / not ready)", worker_id)
        if self._on_removed is not None:
            try:
                self._on_removed(worker_id)
            except Exception:
                logger.exception("on_worker_removed callback failed")
