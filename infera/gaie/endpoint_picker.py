###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Endpoint picking for the Gateway API Inference Extension (GAIE) EPP.

Reuses Infera's existing routing brain — ``WorkerPool`` + ``Policy``
(``KvEventAwarePolicy``) — to choose a worker for a request, mirroring
``AutoRouter``'s PD-preferred selection. Instead of forwarding the request
(the gateway does that), it returns a routing decision: the destination
endpoint plus the headers the gateway/worker need. The policy's in-flight
bookkeeping is kept correct via ``release()``, called when the request ends.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlparse

from infera.common.worker_pool import DisaggMode, WorkerPool
from infera.router.policy.base import Policy

logger = logging.getLogger(__name__)


@dataclass
class PickResult:
    """Routing decision for one request.

    ``destination_endpoint`` is the ``host:port`` the gateway routes to (the
    decode/primary worker). ``*_route_key`` / ``*_blocks`` are opaque handles
    echoed back to the policy in :meth:`EndpointPicker.release` so the
    cost-aware policy can refcount in-flight blocks without re-tokenising.
    """

    destination_endpoint: str
    worker_instance_id: str
    prefill_instance_id: str | None
    route_key: str
    blocks: list[int]
    prefill_route_key: str | None = None
    prefill_blocks: list[int] | None = None


def _endpoint_of(url: str, override_port: int | None = None) -> str:
    """Reduce a worker URL ("http://10.0.0.1:30000") to "host:port" for the
    gateway destination header. ``override_port`` replaces the worker's own
    (engine) port with the InferencePool target port — i.e. the worker pod's
    direct-mode frontend sidecar, which is the actual gateway-routable endpoint
    (the engine port is not a member of the InferencePool). Falls back to the
    raw value when unparseable."""
    parsed = urlparse(url if "://" in url else f"//{url}")
    if parsed.hostname and override_port:
        return f"{parsed.hostname}:{override_port}"
    if parsed.hostname and parsed.port:
        return f"{parsed.hostname}:{parsed.port}"
    if parsed.hostname:
        return parsed.hostname
    return url


class EndpointPicker:
    """PD-preferred endpoint selection over the shared pool + policy.

    Selection mirrors ``AutoRouter``: if a model has BOTH prefill and decode
    workers, route to a decode primary and tag the chosen prefill worker;
    otherwise pick a mixed worker. The policy (kv-aware or round-robin) does
    the actual scoring, so kv-cache locality and load balancing behave exactly
    as in the in-process router.
    """

    def __init__(
        self, pool: WorkerPool, policy: Policy, destination_port: int | None = None
    ) -> None:
        self._pool = pool
        self._policy = policy
        # InferencePool target port (the worker's frontend sidecar). The gateway
        # routes to <pod-ip>:<destination_port>; the worker URL's own port is the
        # engine port, which is not a gateway-routable pool member.
        self._destination_port = destination_port

    def pick(self, model: str | None, body: dict) -> PickResult | None:
        """Choose a worker for ``body``. Returns ``None`` when no active worker
        serves ``model`` (the caller then leaves routing to the gateway/503)."""
        prefill = self._pool.list_active(model=model, mode=DisaggMode.PREFILL)
        decode = self._pool.list_active(model=model, mode=DisaggMode.DECODE)
        if prefill and decode:
            return self._pick_disagg(prefill, decode, body)
        mixed = self._pool.list_active(model=model, mode=DisaggMode.MIXED)
        if mixed:
            return self._pick_single(mixed, body, role_hint=None)
        return None

    def _pick_single(self, candidates, body: dict, *, role_hint: str | None) -> PickResult:
        target, blocks = self._policy.pick(candidates, body, role_hint=role_hint)
        self._policy.on_request_started(target.route_key, blocks)
        return PickResult(
            destination_endpoint=_endpoint_of(target.worker.url, self._destination_port),
            worker_instance_id=target.worker.worker_id,
            prefill_instance_id=None,
            route_key=target.route_key,
            blocks=blocks,
        )

    def _pick_disagg(self, prefill, decode, body: dict) -> PickResult:
        # Decode is the primary endpoint the gateway routes to; the decode
        # worker coordinates prefill via the x-prefill-instance-id header.
        d_target, d_blocks = self._policy.pick(decode, body, role_hint="decode")
        p_target, p_blocks = self._policy.pick(prefill, body, role_hint="prefill")
        self._policy.on_request_started(d_target.route_key, d_blocks)
        self._policy.on_request_started(p_target.route_key, p_blocks)
        return PickResult(
            destination_endpoint=_endpoint_of(d_target.worker.url, self._destination_port),
            worker_instance_id=d_target.worker.worker_id,
            prefill_instance_id=p_target.worker.worker_id,
            route_key=d_target.route_key,
            blocks=d_blocks,
            prefill_route_key=p_target.route_key,
            prefill_blocks=p_blocks,
        )

    def release(self, result: PickResult) -> None:
        """Release the policy in-flight bookkeeping once the request finishes."""
        self._policy.on_request_finished(result.route_key, result.blocks)
        if result.prefill_route_key is not None:
            self._policy.on_request_finished(result.prefill_route_key, result.prefill_blocks)
