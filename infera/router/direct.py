###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import logging

from fastapi import Response
from fastapi.responses import JSONResponse

from infera.router.cache_control import parse_cache_hints
from infera.router.disagg import DisaggRouter
from infera.router.mixed import MixedRouter, _Retry
from infera.router.policy.target import RouteTarget
from infera.server import metrics

logger = logging.getLogger(__name__)

# Body annotations carrying the EPP-selected worker ids, set by the server from
# the GAIE header-routing hints:
#   x-worker-instance-id  -> DIRECT_WORKER_KEY  (decode / primary worker)
#   x-prefill-instance-id -> DIRECT_PREFILL_KEY (prefill worker; disagg only)
DIRECT_WORKER_KEY = "_infera_direct_worker"
DIRECT_PREFILL_KEY = "_infera_direct_prefill"


class DirectRouter(MixedRouter):
    """Router-mode ``direct``: dispatch to the worker the gateway already
    chose, identified by ``x-worker-instance-id`` (stashed on the body as
    ``_infera_direct_worker``). Selection happened in the GAIE EPP, so this
    router never calls ``policy.pick`` — it looks the worker up in the pool and
    reuses :class:`MixedRouter`'s transport (HTTP / per-instance NATS, streaming
    and unary alike).

    There is no failover here: the gateway/EPP owns worker selection, so a
    failed attempt is surfaced to the client rather than silently re-routed
    (re-routing would contradict the gateway's decision and double-bill the
    policy). When the header is absent (e.g. the frontend is hit directly,
    without a gateway in front), it falls back to the normal cost-aware
    :class:`MixedRouter` selection so the same binary still works standalone.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Disaggregated leg for the PD direct path. Shares pool + policy +
        # nats_client so a gateway that picked both a decode and a prefill
        # worker runs the full dual-dispatch (concurrent / serial-pull,
        # HTTP / NATS) without re-selecting.
        self._disagg = DisaggRouter(self.pool, self.policy, nats_client=self.nats_client)

    async def aclose(self) -> None:
        await super().aclose()
        await self._disagg.aclose()

    async def dispatch(
        self,
        body: dict,
        *,
        stream: bool,
        path: str = "/v1/chat/completions",
    ) -> Response:
        worker_id = body.pop(DIRECT_WORKER_KEY, None)
        prefill_id = body.pop(DIRECT_PREFILL_KEY, None)
        if not worker_id:
            # No EPP hint -> behave like a normal mixed router (standalone use).
            logger.debug("direct: no %s on request; falling back to policy pick", DIRECT_WORKER_KEY)
            return await super().dispatch(body, stream=stream, path=path)

        # PD direct: the gateway chose decode (worker_id) + prefill; run the
        # dual-dispatch against exactly those two workers.
        if prefill_id:
            return await self._disagg.dispatch_direct(
                body,
                stream=stream,
                path=path,
                prefill_id=prefill_id,
                decode_id=worker_id,
            )

        with metrics.track_request(router="direct") as obs:
            worker = self.pool.get(worker_id)
            if worker is None:
                obs["outcome"] = "503"
                logger.warning("direct: worker %r from header not in pool", worker_id)
                return JSONResponse(
                    content={"error": f"worker {worker_id!r} not found (stale gateway routing?)"},
                    status_code=503,
                )

            hints = body.get("_infera_cache_hints") or parse_cache_hints(body)
            # No DP-rank steering in direct mode: the gateway addressed a worker
            # endpoint, the engine's own controller fans across its ranks.
            target = RouteTarget(worker)
            try:
                return await self._attempt(target, [], body, hints, path, stream, obs)
            except _Retry as r:
                # Selection is the gateway's job; surface the error as-is.
                return r.response
