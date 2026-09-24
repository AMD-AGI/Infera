###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Gate PD prefill registration on a decode worker, not engine start.

Prefill and decode load weights in parallel. SGLang's PD warmup is a
``/generate`` to bootstrap host 2.2.2.2 during FastAPI startup; that fake
session stays inflight on Mooncake and later real KV transfers fail. The
barrier therefore passes ``--skip-server-warmup`` into launch_server and
waits for decode to register (``/health`` 200 + worker-info) before this
process advertises itself. It does not replay the fake warmup; the first
real request owns the Mooncake session.

Two things make the record alone insufficient, and both are handled below:

* A container restart that skips the SIGTERM handler never clears the
  annotation (see :mod:`infera.common.discovery_k8s`), so a decode that is
  reloading weights still advertises the previous process. The Pod's ``Ready``
  condition is checked alongside the annotation: a restart drops it until the
  new process passes its startup probe.
* An unscoped Pod list would accept a decode from a *different* deployment
  that happens to serve the same model, which is not a Mooncake peer. The
  label selector must resolve to something, or this module refuses to gate.
  The etcd path has no equivalent label: isolation is ``--etcd-prefix``,
  which must be unique per deployment. A shared default prefix can match
  another deployment's decode worker.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
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

SKIP_SERVER_WARMUP_FLAG = "--skip-server-warmup"
DEFAULT_PD_PROBE_TIMEOUT = 300.0
DEFAULT_PD_PROBE_ATTEMPTS = 3
DEFAULT_PD_PROBE_RETRY_SLEEP = 35.0

# Every Pod of an InferaDeployment carries this (operator: builders.go
# labelKeyDeployment), which is what scopes the list to real Mooncake peers.
DEPLOYMENT_LABEL = "infera.amd.com/deployment"

UNSCOPED_BARRIER_ERROR = (
    "cannot scope the decode barrier: this Pod carries no "
    f"{DEPLOYMENT_LABEL} label and neither INFERA_K8S_LABEL_SELECTOR nor "
    "WORKLOAD_ID is set. Pass --k8s-label-selector, or --no-wait-for-decode "
    "to start without the barrier."
)

# Decode legs of this size load for tens of minutes; the budget is deliberately
# generous because the alternative to waiting is a poisoned session cache.
DEFAULT_DECODE_READY_TIMEOUT = 14400.0

ListWorkers = Callable[[], Awaitable[list[dict[str, Any]]]]


class K8sLabelLookupError(RuntimeError):
    """A transient failure reading this Pod from the Kubernetes API."""


#: Failures of a reachable-but-failing discovery lookup. The decode skips its
#: prefill probe on these rather than failing a worker that has loaded weights.
DISCOVERY_LOOKUP_ERRORS: tuple[type[BaseException], ...] = (
    OSError,
    httpx.HTTPError,
    asyncio.TimeoutError,
    ValueError,
    KeyError,
    K8sLabelLookupError,
)

#: Failures of one PD peer probe: a failed transfer, a transport error, or the
#: probe budget running out.
PEER_PROBE_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    OSError,
    httpx.HTTPError,
    asyncio.TimeoutError,
)


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


def pd_probe_reserve_seconds(total_timeout: float) -> float:
    """Slice of the decode-ready budget held back for the KV probe.

    Discovery (selector plus worker lookup) can otherwise poll until the shared
    deadline and hand the probe a budget of zero, turning a healthy peer into a
    timeout. The reserve is capped by the total so it can never exceed it.
    """
    return min(max(0.0, float(total_timeout)), DEFAULT_PD_PROBE_TIMEOUT)


def discovery_budget_seconds(total_timeout: float) -> float:
    """Decode-ready budget left for discovery once the probe reserve is taken.

    Zero means discovery still runs its one immediate lookup, but never sleeps.
    """
    return max(0.0, float(total_timeout) - pd_probe_reserve_seconds(total_timeout))


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
    retries: int = 3,
    retry_sleep: float = 1.0,
) -> str:
    """Label selector scoping the peer list to this deployment's workers.

    Flag, then INFERA_K8S_LABEL_SELECTOR, then this Pod's own
    ``infera.amd.com/deployment`` label, then WORKLOAD_ID (set by SaFE, not by
    the operator). A failed GET of this Pod is retried and then raised -- it
    must not fall through to WORKLOAD_ID, which can name a different
    deployment. Raises when none of the sources resolve.
    """
    if explicit:
        return explicit
    env = os.environ.get("INFERA_K8S_LABEL_SELECTOR")
    if env:
        return env

    own = await own_pod_deployment_label(
        namespace=namespace,
        pod_name=pod_name,
        http=http,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    if own:
        return f"{DEPLOYMENT_LABEL}={own}"

    workload_id = os.environ.get("WORKLOAD_ID")
    if workload_id:
        return f"{DEPLOYMENT_LABEL}={workload_id}"

    raise RuntimeError(UNSCOPED_BARRIER_ERROR)


def ensure_k8s_label_selector_source(explicit: str | None) -> None:
    """Refuse a barrier that can never scope itself, before the weight load.

    Only the sources readable without the apiserver are decided here: a
    ``POD_NAME`` leaves the answer to :func:`resolve_k8s_label_selector`,
    which is the only place that can see the Pod's own label. Raising there
    instead costs a full weight load per restart, since the barrier runs
    after ``engine.start()``.
    """
    if explicit or os.environ.get("INFERA_K8S_LABEL_SELECTOR"):
        return
    if os.environ.get("POD_NAME") or os.environ.get("WORKLOAD_ID"):
        return
    raise RuntimeError(UNSCOPED_BARRIER_ERROR)


def ensure_barrier_discovery_is_reachable(
    discovery_backend: str,
    *,
    k8s_label_selector: str | None,
    etcd_endpoint: str | None,
) -> None:
    """Refuse a barrier whose discovery cannot resolve, before the weight load.

    Both backends are decided here for the same reason: the barrier itself runs
    after ``engine.start()``, so a config it can never satisfy would otherwise
    cost a full weight load on every restart before saying so.
    """
    if discovery_backend == "kubernetes":
        ensure_k8s_label_selector_source(k8s_label_selector)
        return
    if not etcd_endpoint:
        raise RuntimeError(
            "--wait-for-decode with --discovery-backend=etcd requires --etcd-endpoint"
        )


async def own_pod_deployment_label(
    *,
    namespace: str | None = None,
    pod_name: str | None = None,
    http: httpx.AsyncClient | None = None,
    retries: int = 3,
    retry_sleep: float = 1.0,
) -> str | None:
    """Read this Pod's deployment label (the operator stamps it on every role).

    A missing Pod name is not an error (caller may fall back to WORKLOAD_ID).
    A failed GET after retries is: swallowing it would drop through to a
    selector that does not match this deployment.
    """
    name = pod_name if pod_name is not None else os.environ.get("POD_NAME", "")
    if not name:
        return None
    ns = k8s_namespace(namespace)
    owns_client = http is None
    client = http if http is not None else make_client(timeout=10.0)
    last_exc: Exception | None = None
    try:
        attempts = max(1, retries)
        for attempt in range(attempts):
            try:
                if not owns_client:
                    refresh_k8s_auth(client)
                resp = await client.get(f"/api/v1/namespaces/{ns}/pods/{name}")
                resp.raise_for_status()
                labels = ((resp.json().get("metadata") or {}).get("labels")) or {}
                return labels.get(DEPLOYMENT_LABEL) or None
            except Exception as exc:  # noqa: BLE001 - retried below
                last_exc = exc
                logger.warning(
                    "could not read this Pod's labels (%s/%s, attempt %d/%d): %s",
                    ns,
                    name,
                    attempt + 1,
                    attempts,
                    exc,
                )
                if attempt + 1 < attempts:
                    await asyncio.sleep(retry_sleep)
        raise K8sLabelLookupError(
            f"could not read this Pod's labels ({ns}/{name}) after {attempts} attempts"
        ) from last_exc
    finally:
        if owns_client:
            await client.aclose()


async def wait_for_k8s_label_selector(
    explicit: str | None,
    *,
    namespace: str | None = None,
    pod_name: str | None = None,
    http: httpx.AsyncClient | None = None,
    timeout: float,
    poll_interval: float = 5.0,
) -> str:
    """Resolve a scoped selector while tolerating transient API failures."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            return await resolve_k8s_label_selector(
                explicit,
                namespace=namespace,
                pod_name=pod_name,
                http=http,
                retries=1,
            )
        except K8sLabelLookupError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            await asyncio.sleep(min(poll_interval, remaining))


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


def is_compatible_prefill_worker(
    payload: dict[str, Any],
    *,
    model_name: str,
    engine: str = EngineType.SGLANG.value,
    protocol: str = SGLANG_BOOTSTRAP_PROTOCOL,
) -> bool:
    """True when a registration payload is a matching PD prefill worker.

    The mirror of :func:`is_compatible_decode_worker`, and additionally
    requires a bootstrap address: a decode verifying this peer has to dial it,
    and a prefill that advertised none cannot be probed at all.
    """
    if not payload:
        return False
    if str(payload.get("disagg_mode") or "") != DisaggMode.PREFILL.value:
        return False
    if str(payload.get("model_name") or "") != model_name:
        return False
    if str(payload.get("engine") or EngineType.SGLANG.value) != engine:
        return False
    meta = payload.get("disagg_meta") or {}
    if not isinstance(meta, dict):
        return False
    if str(meta.get("protocol") or "") != protocol:
        return False
    return prefill_bootstrap_addr(payload) is not None


def prefill_bootstrap_addr(payload: dict[str, Any]) -> tuple[str, int] | None:
    """Parse ``host, port`` out of a prefill registration's disagg_meta.

    Split from the right so an IPv6 literal keeps its colons. Returns None for
    anything unparseable rather than raising, so a single malformed
    registration does not stop the caller from considering other peers.
    """
    meta = payload.get("disagg_meta") or {}
    if not isinstance(meta, dict):
        return None
    params = meta.get("params") or {}
    if not isinstance(params, dict):
        return None
    raw = str(params.get("bootstrap_addr") or "")
    host, sep, port = raw.rpartition(":")
    if not sep or not host:
        return None
    try:
        return host, int(port)
    except ValueError:
        return None


def should_wait_for_decode(
    disaggregation_mode: str | None,
    wait_for_decode: bool | None,
) -> bool:
    """Prefill waits by default; --no-wait-for-decode opts out."""
    if str(disaggregation_mode or "") != "prefill":
        return False
    return wait_for_decode is not False


def should_verify_prefill(
    disaggregation_mode: str | None,
    wait_for_decode: bool | None,
) -> bool:
    """Decode verifies a registered prefill by default; the same flag opts out.

    This is the reverse of the prefill barrier and deliberately never waits:
    if both legs blocked on each other a fresh deployment could never start.
    """
    if str(disaggregation_mode or "") != DisaggMode.DECODE.value:
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
    """List worker registration payloads under an etcd prefix.

    Unlike the Kubernetes path, this listing is not deployment-scoped.
    Operators must give each deployment its own ``--etcd-prefix``.
    """
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

    One lookup always runs, even on an exhausted budget: callers share their
    deadline with earlier steps, and a decode that is already registered must
    not be reported as missing. The budget still bounds every sleep.
    """
    sleeper = sleep or asyncio.sleep
    deadline = time.monotonic() + timeout
    last_log = 0.0
    started = time.monotonic()
    while True:
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


def ensure_skip_server_warmup(argv: list[str]) -> list[str]:
    """Append --skip-server-warmup so launch_server does not PD-warmup itself."""
    if SKIP_SERVER_WARMUP_FLAG in argv:
        return argv
    return [*argv, SKIP_SERVER_WARMUP_FLAG]


def apply_pd_probe_recovery_defaults(
    disaggregation_mode: str | None,
    transfer_backend: str | None,
) -> dict[str, str]:
    """Enable Mooncake session recovery required by delayed probe retries."""
    if disaggregation_mode != "prefill" or transfer_backend != "mooncake":
        return {}
    defaults = {
        "SGLANG_ENABLE_FAILED_SESSION_PROBE": "1",
        "SGLANG_FAILED_SESSION_PROBE_INTERVAL_S": "5",
    }
    applied: dict[str, str] = {}
    for key, value in defaults.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
    return applied


def pd_peer_probe_payload(
    *,
    bootstrap_host: str,
    bootstrap_port: int,
    room: int,
    dp_rank: int,
    prefill_dp_rank: int | None = None,
) -> dict[str, Any]:
    """Build a real-peer SGLang request that verifies KV transfer readiness.

    ``prefill_dp_rank`` names the prefill rank holding the KV for this room and
    belongs on the decode leg only: the decode scheduler needs the producer's
    rank, which is not derivable from its own ``routed_dp_rank`` once the two
    legs run different DP sizes.
    """
    payload: dict[str, Any] = {
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "bootstrap_host": bootstrap_host,
        "bootstrap_port": int(bootstrap_port),
        "bootstrap_room": room,
        "input_ids": [10, 11, 12, 13],
        "routed_dp_rank": dp_rank,
        "rid": f"infera-probe-{room}",
    }
    if prefill_dp_rank is not None:
        payload["disagg_prefill_dp_rank"] = int(prefill_dp_rank)
    return payload


#: Bound on releasing a probe room, so a wedged engine cannot hold up a probe
#: that has already failed or been cancelled.
_PROBE_ABORT_TIMEOUT_S = 10.0


async def _abort_probe_room(client: Any, prefill_url: str, decode_url: str, rid: str) -> None:
    """Abort a probe request on both engines, so no Mooncake session is left open.

    Best effort: failures are ignored, since the probe outcome is already decided.
    """
    body = {"rid": rid}
    try:
        await asyncio.wait_for(
            asyncio.gather(
                client.post(f"{prefill_url.rstrip('/')}/abort_request", json=body),
                client.post(f"{decode_url.rstrip('/')}/abort_request", json=body),
                return_exceptions=True,
            ),
            _PROBE_ABORT_TIMEOUT_S,
        )
    except Exception:  # noqa: BLE001 - best effort, the outcome is already decided
        logger.warning("decode barrier: could not abort probe room %s", rid)


def _probe_failure_details(failed: list[Any]) -> str:
    return ", ".join(
        f"{type(result).__name__}: {result}"
        if isinstance(result, BaseException)
        else f"HTTP {result.status_code}: {result.text[:200]}"
        for result in failed
    )


async def probe_until_one_passes(
    probes: list[tuple[str, Callable[[float], Awaitable[None]]]],
    *,
    deadline: float,
    min_budget: float,
    clock: Callable[[], float],
) -> bool:
    """Run peer probes in order until one passes.

    Each probe receives the budget left and is bounded by it. Returns True once
    a probe passes, and False when the budget ran out before any probe ran.
    When every probe that ran failed, the last failure is raised, so a broken
    KV path still stops registration while a single dead peer does not.
    """
    last_exc: BaseException | None = None
    for name, probe in probes:
        remaining = deadline - clock()
        if remaining < min_budget:
            break
        try:
            await asyncio.wait_for(probe(remaining), timeout=remaining)
            return True
        except PEER_PROBE_ERRORS as exc:
            logger.warning("prefill probe: %s failed: %s", name, exc)
            last_exc = exc
    if last_exc is not None:
        raise last_exc
    return False


async def verify_pd_peer(
    *,
    prefill_url: str,
    decode_url: str,
    bootstrap_host: str,
    bootstrap_port: int,
    dp_size: int = 1,
    decode_dp_size: int = 1,
    timeout: float = DEFAULT_PD_PROBE_TIMEOUT,
    attempts: int = DEFAULT_PD_PROBE_ATTEMPTS,
    retry_sleep: float = DEFAULT_PD_PROBE_RETRY_SLEEP,
    room_seed: int | None = None,
    http: httpx.AsyncClient | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
) -> None:
    """Transfer one real KV block per DP rank before advertising the prefill.

    Probes cover ``max(dp_size, decode_dp_size)`` pairs so neither leg keeps an
    unexercised rank, and each endpoint is addressed with its own local rank.
    The bootstrap room stays aligned to the producing prefill rank and is
    unique per attempt.

    A failed transfer aborts both legs and retries with a new room so a
    transient RDMA/Mooncake error does not permanently block registration.
    The first probe is issued immediately; retry_sleep applies only after abort.
    """
    ranks = max(1, int(dp_size))
    decode_ranks = max(1, int(decode_dp_size))
    pairs = max(ranks, decode_ranks)
    tries = max(1, int(attempts))
    sleeper = sleep or asyncio.sleep
    base_room = room_seed if room_seed is not None else secrets.randbits(63)
    base_room -= base_room % ranks
    # A multiple of ranks, so a retry moves to a fresh room without breaking
    # the room -> prefill rank alignment.
    attempt_stride = pairs * ranks
    owns_client = http is None
    client = http if http is not None else httpx.AsyncClient(timeout=timeout)
    try:
        for pair in range(pairs):
            dp_rank = pair % ranks
            decode_dp_rank = pair % decode_ranks
            last_details = ""
            for attempt in range(tries):
                room = base_room + pair + attempt * attempt_stride
                prefill_body = pd_peer_probe_payload(
                    bootstrap_host=bootstrap_host,
                    bootstrap_port=bootstrap_port,
                    room=room,
                    dp_rank=dp_rank,
                )
                # routed_dp_rank is local to each endpoint. The bootstrap room
                # remains aligned to the producing prefill rank, which decode
                # is told explicitly so it does not assume its own rank.
                decode_body = pd_peer_probe_payload(
                    bootstrap_host=bootstrap_host,
                    bootstrap_port=bootstrap_port,
                    room=room,
                    dp_rank=decode_dp_rank,
                    prefill_dp_rank=dp_rank,
                )
                # An injected client carries the caller's timeout; the probe
                # budget is passed per request so it is the one that applies.
                try:
                    results = await asyncio.gather(
                        client.post(
                            f"{prefill_url.rstrip('/')}/generate",
                            json=prefill_body,
                            timeout=timeout,
                        ),
                        client.post(
                            f"{decode_url.rstrip('/')}/generate",
                            json=decode_body,
                            timeout=timeout,
                        ),
                        return_exceptions=True,
                    )
                except asyncio.CancelledError:
                    # A probe cut short by its caller's budget leaves the room
                    # open on both engines; release it before propagating.
                    await _abort_probe_room(client, prefill_url, decode_url, prefill_body["rid"])
                    raise
                failed = [
                    result
                    for result in results
                    if isinstance(result, BaseException) or result.status_code >= 400
                ]
                if not failed:
                    break

                await _abort_probe_room(client, prefill_url, decode_url, prefill_body["rid"])
                last_details = _probe_failure_details(failed)
                if attempt + 1 >= tries:
                    raise RuntimeError(
                        f"PD peer verification failed for prefill dp_rank={dp_rank} "
                        f"/ decode dp_rank={decode_dp_rank} "
                        f"after {tries} attempts: {last_details}"
                    )
                logger.warning(
                    "decode barrier: PD peer probe failed for dp_rank=%d "
                    "attempt %d/%d; aborting and retrying: %s",
                    dp_rank,
                    attempt + 1,
                    tries,
                    last_details,
                )
                if retry_sleep > 0:
                    await sleeper(retry_sleep)
        logger.info(
            "decode barrier: verified real KV transfer to %s for %d DP rank pair(s)",
            decode_url,
            pairs,
        )
    finally:
        if owns_client:
            await client.aclose()
