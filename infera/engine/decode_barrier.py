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

Two things make the record alone insufficient, and both are handled below:

* A container restart that skips the SIGTERM handler never clears the
  annotation (see :mod:`infera.common.discovery_k8s`), so a decode that is
  reloading weights still advertises the previous process. The Pod's ``Ready``
  condition is checked alongside the annotation: a restart drops it until the
  new process passes its startup probe.
* An unscoped Pod list would accept a decode from a *different* deployment
  that happens to serve the same model, which is not a Mooncake peer. The
  label selector must resolve to something, or this module refuses to gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
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
from infera.common.k8s_client import _read_token, in_cluster_namespace, make_client
from infera.common.worker_pool import DisaggMode, EngineType

logger = logging.getLogger(__name__)

SGLANG_BOOTSTRAP_PROTOCOL = "sglang-bootstrap"

# Every Pod of an InferaDeployment carries this (operator: builders.go
# labelKeyDeployment), which is what scopes the list to real Mooncake peers.
DEPLOYMENT_LABEL = "infera.amd.com/deployment"

# Decode legs of this size load for tens of minutes; the budget is deliberately
# generous because the alternative to waiting is a poisoned session cache.
DEFAULT_DECODE_READY_TIMEOUT = 14400.0

ListWorkers = Callable[[], Awaitable[list[dict[str, Any]]]]


def decode_ready_timeout_seconds(explicit: float | None) -> float:
    """Resolve the decode-wait budget: flag, then env, then the default.

    Deliberately NOT falling back to INFERA_ENGINE_READY_TIMEOUT: that is the
    engine's own /health deadline, which recipes raise for slow weight loads.
    Reading it here would silently retune this barrier for an unrelated reason.
    """
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get("INFERA_DECODE_READY_TIMEOUT")
    if raw:
        try:
            return float(raw)
        except ValueError:
            logger.warning(
                "INFERA_DECODE_READY_TIMEOUT=%r is not a number; using %.0fs",
                raw,
                DEFAULT_DECODE_READY_TIMEOUT,
            )
    return DEFAULT_DECODE_READY_TIMEOUT


def refresh_k8s_auth(client: httpx.AsyncClient) -> None:
    """Re-read the mounted ServiceAccount token onto an existing client.

    The barrier can poll for hours on one client, and kubelet rotates the
    projected token well inside that window; a header captured at construction
    time starts coming back 401.
    """
    try:
        client.headers["Authorization"] = f"Bearer {_read_token()}"
    except OSError as exc:
        logger.warning("could not re-read the ServiceAccount token: %s", exc)


def k8s_namespace(explicit: str | None = None) -> str:
    """Namespace for peer lookups: flag, POD_NAMESPACE, then the mounted SA."""
    return explicit or os.environ.get("POD_NAMESPACE") or in_cluster_namespace()


async def resolve_k8s_label_selector(
    explicit: str | None,
    *,
    namespace: str | None = None,
    pod_name: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> str:
    """Label selector scoping the peer list to this deployment's workers.

    Flag, then INFERA_K8S_LABEL_SELECTOR, then this Pod's own
    ``infera.amd.com/deployment`` label, then WORKLOAD_ID (set by SaFE, not by
    the operator). Raises when none of them resolve: listing the whole
    namespace would accept a decode belonging to another deployment, which is
    a barrier that reports success without having gated anything.
    """
    if explicit:
        return explicit
    env = os.environ.get("INFERA_K8S_LABEL_SELECTOR")
    if env:
        return env

    own = await own_pod_deployment_label(namespace=namespace, pod_name=pod_name, http=http)
    if own:
        return f"{DEPLOYMENT_LABEL}={own}"

    workload_id = os.environ.get("WORKLOAD_ID")
    if workload_id:
        return f"{DEPLOYMENT_LABEL}={workload_id}"

    raise RuntimeError(
        "cannot scope the decode barrier: this Pod carries no "
        f"{DEPLOYMENT_LABEL} label and neither INFERA_K8S_LABEL_SELECTOR nor "
        "WORKLOAD_ID is set. Pass --k8s-label-selector, or --no-wait-for-decode "
        "to start without the barrier."
    )


async def own_pod_deployment_label(
    *,
    namespace: str | None = None,
    pod_name: str | None = None,
    http: httpx.AsyncClient | None = None,
) -> str | None:
    """Read this Pod's deployment label (the operator stamps it on every role)."""
    name = pod_name if pod_name is not None else os.environ.get("POD_NAME", "")
    if not name:
        return None
    ns = k8s_namespace(namespace)
    owns_client = http is None
    client = http if http is not None else make_client(timeout=10.0)
    try:
        resp = await client.get(f"/api/v1/namespaces/{ns}/pods/{name}")
        resp.raise_for_status()
        labels = ((resp.json().get("metadata") or {}).get("labels")) or {}
    except Exception as exc:  # noqa: BLE001 - fall through to the other sources
        logger.warning("could not read this Pod's labels (%s/%s): %s", ns, name, exc)
        return None
    finally:
        if owns_client:
            await client.aclose()
    return labels.get(DEPLOYMENT_LABEL) or None


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


def _pod_is_ready(pod: dict[str, Any]) -> bool:
    """Whether the kubelet currently reports the Pod as Ready.

    This is what separates a live decode from one whose container restarted
    and left its annotation behind: the condition goes False for the whole of
    the replacement process's startup probe.
    """
    for cond in (pod.get("status") or {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return str(cond.get("status")) == "True"
    return False


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
    if not _pod_is_ready(pod):
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
    """List worker-info annotations from Ready Pods in the namespace."""
    ns = k8s_namespace(namespace)
    skip = skip_pod_name if skip_pod_name is not None else os.environ.get("POD_NAME", "")
    params: dict[str, str] = {}
    if label_selector:
        params["labelSelector"] = label_selector
    owns_client = http is None
    client = http if http is not None else make_client(timeout=10.0)
    try:
        if not owns_client:
            refresh_k8s_auth(client)
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
    """Poll until a compatible decode worker is registered, or time out.

    A failed lookup is retried rather than raised. This runs before
    ``engine.start()`` for hours at a time, so one transient apiserver or etcd
    error would otherwise kill a prefill worker that has nothing wrong with
    it; the deadline is the only thing that gives up.
    """
    sleeper = sleep or asyncio.sleep
    deadline = time.monotonic() + timeout
    last_log = 0.0
    started = time.monotonic()
    while time.monotonic() < deadline:
        try:
            workers = await list_workers()
        except Exception as exc:  # noqa: BLE001 - transient lookup failures are retried
            workers = []
            logger.warning("decode barrier: worker lookup failed (retrying): %s", exc)
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
