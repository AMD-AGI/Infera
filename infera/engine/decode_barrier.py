###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Hold a PD prefill worker until a compatible decode worker has registered.

SGLang PD warmup issues a real KV transfer. If prefill reaches warmup before
decode is listening on Mooncake, that request stays inflight and poisons the
session cache even after decode later becomes ready. Decode registers its
worker-info annotation only after ``/health`` is 200, so waiting on that
record is the barrier.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from infera.common.discovery import (
    DEFAULT_PREFIX,
    _b64,
    _b64bytes,
    _normalize_endpoint,
    _range_end_for_prefix,
    _unb64,
)
from infera.common.discovery_k8s import WORKER_INFO_ANNOTATION
from infera.common.k8s_client import in_cluster_namespace, make_client
from infera.common.worker_pool import DisaggMode, EngineType

logger = logging.getLogger(__name__)

SGLANG_BOOTSTRAP_PROTOCOL = "sglang-bootstrap"

ListWorkers = Callable[[], Awaitable[list[dict[str, Any]]]]


def decode_ready_timeout_seconds(explicit: float | None) -> float:
    """Resolve the decode-wait budget: flag, then env, then 14400s."""
    if explicit is not None:
        return float(explicit)
    for key in ("INFERA_DECODE_READY_TIMEOUT", "INFERA_ENGINE_READY_TIMEOUT"):
        raw = os.environ.get(key)
        if raw:
            return float(raw)
    return 14400.0


def resolve_k8s_label_selector(explicit: str | None) -> str | None:
    """Label selector for listing peer engine Pods.

    Operator flag / INFERA_K8S_LABEL_SELECTOR first. SaFE/Infera workloads
    stamp infera.amd.com/deployment=$WORKLOAD_ID on every role, which is
    enough to find the decode Pod in the same deployment.
    """
    if explicit:
        return explicit
    env = os.environ.get("INFERA_K8S_LABEL_SELECTOR")
    if env:
        return env
    workload_id = os.environ.get("WORKLOAD_ID")
    if workload_id:
        return f"infera.amd.com/deployment={workload_id}"
    return None


def is_compatible_decode_worker(
    payload: dict[str, Any],
    *,
    model_name: str,
    engine: str = EngineType.SGLANG.value,
    protocol: str = SGLANG_BOOTSTRAP_PROTOCOL,
) -> bool:
    """True when a registration payload is a matching PD decode worker."""
    if not payload:
        return False
    if str(payload.get("disagg_mode") or "") != DisaggMode.DECODE.value:
        return False
    if str(payload.get("model_name") or "") != model_name:
        return False
    if str(payload.get("engine") or EngineType.SGLANG.value) != engine:
        return False
    meta = payload.get("disagg_meta") or {}
    if not isinstance(meta, dict):
        return False
    # Missing protocol is not a match: a decode worker that did not advertise
    # sglang-bootstrap is not a safe Mooncake peer.
    return str(meta.get("protocol") or "") == protocol


def should_wait_for_decode(
    disaggregation_mode: str | None,
    wait_for_decode: bool | None,
) -> bool:
    """Prefill waits by default; --no-wait-for-decode opts out."""
    if str(disaggregation_mode or "") != "prefill":
        return False
    return wait_for_decode is not False


def _pod_is_listable_worker(pod: dict[str, Any], *, skip_name: str) -> dict[str, Any] | None:
    meta = pod.get("metadata") or {}
    name = meta.get("name") or ""
    if skip_name and name == skip_name:
        return None
    if meta.get("deletionTimestamp"):
        return None
    phase = ((pod.get("status") or {}).get("phase")) or ""
    if phase != "Running":
        return None
    raw = (meta.get("annotations") or {}).get(WORKER_INFO_ANNOTATION)
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


async def list_k8s_worker_payloads(
    *,
    namespace: str | None = None,
    label_selector: str | None = None,
    skip_pod_name: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """List worker-info annotations from Running Pods in the namespace."""
    ns = namespace or in_cluster_namespace()
    skip = skip_pod_name if skip_pod_name is not None else os.environ.get("POD_NAME", "")
    params: dict[str, str] = {}
    if label_selector:
        params["labelSelector"] = label_selector
    owns_client = http is None
    client = http if http is not None else make_client(timeout=10.0)
    try:
        resp = await client.get(f"/api/v1/namespaces/{ns}/pods", params=params)
        resp.raise_for_status()
        body = resp.json()
    finally:
        if owns_client:
            await client.aclose()
    out: list[dict[str, Any]] = []
    for pod in body.get("items") or []:
        payload = _pod_is_listable_worker(pod, skip_name=skip)
        if payload is not None:
            out.append(payload)
    return out


async def list_etcd_worker_payloads(
    endpoint: str,
    prefix: str = DEFAULT_PREFIX,
    *,
    http: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """List worker registration payloads under an etcd prefix."""
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    base = _normalize_endpoint(endpoint)
    owns_client = http is None
    client = http if http is not None else httpx.AsyncClient(base_url=base, timeout=10.0)
    try:
        r = await client.post(
            "/v3/kv/range",
            json={
                "key": _b64(prefix),
                "range_end": _b64bytes(_range_end_for_prefix(prefix)),
            },
        )
        r.raise_for_status()
        kvs = r.json().get("kvs") or []
    finally:
        if owns_client:
            await client.aclose()
    out: list[dict[str, Any]] = []
    for kv in kvs:
        raw = kv.get("value")
        if not raw:
            continue
        try:
            payload = json.loads(_unb64(raw))
        except (TypeError, json.JSONDecodeError, ValueError):
            continue
        if isinstance(payload, dict):
            out.append(payload)
    return out


async def wait_for_decode(
    list_workers: ListWorkers,
    *,
    model_name: str,
    engine: str = EngineType.SGLANG.value,
    protocol: str = SGLANG_BOOTSTRAP_PROTOCOL,
    timeout: float,
    poll_interval: float = 5.0,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> dict[str, Any]:
    """Poll until a compatible decode worker is registered, or time out."""
    import asyncio
    import time

    sleeper = sleep or asyncio.sleep
    deadline = time.monotonic() + timeout
    last_log = 0.0
    started = time.monotonic()
    while time.monotonic() < deadline:
        workers = await list_workers()
        for payload in workers:
            if is_compatible_decode_worker(
                payload, model_name=model_name, engine=engine, protocol=protocol
            ):
                logger.info(
                    "decode barrier: found decode worker %s for model %s",
                    payload.get("worker_id") or payload.get("url"),
                    model_name,
                )
                return payload
        now = time.monotonic()
        if now - last_log >= 30.0:
            logger.info(
                "decode barrier: waiting for a registered %s decode worker "
                "(model=%s, protocol=%s, elapsed=%.0fs)",
                engine,
                model_name,
                protocol,
                now - started,
            )
            last_log = now
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        await sleeper(min(poll_interval, remaining))
    raise TimeoutError(
        f"no compatible {engine} decode worker registered for model {model_name!r} "
        f"(protocol={protocol}) after {timeout:.0f}s; prefill warmup would "
        "poison Mooncake if started now"
    )
