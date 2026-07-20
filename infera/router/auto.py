###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from fastapi import Response

from infera.common.worker_pool import DisaggMode
from infera.router.base import BaseRouter
from infera.router.disagg import DisaggRouter
from infera.router.mixed import MixedRouter


class AutoRouter(BaseRouter):
    """Per-request router selector.

    Selection policy (v0.1: PD-preferred with mixed fallback):
      - If the model has BOTH prefill and decode workers     → DisaggRouter
      - Otherwise (only mixed, partial PD, or empty)         → MixedRouter
        (MixedRouter itself returns 503 if no mixed worker is available)

    This supports mixed deployments (some models PD, others mixed) and rolling
    upgrades without a server restart. The two inner routers share the pool
    and policy, so any state held by the policy (e.g. round-robin counters)
    is fleet-wide.

    Future evolution: smarter selection (e.g. selective PD by prompt length /
    prefix-cache hit) belongs in the Policy layer, not here. This router stays
    a dumb dispatcher.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mixed = MixedRouter(
            self.pool,
            self.policy,
            nats_client=self.nats_client,
            request_max_retries=self.request_max_retries,
        )
        # Pass the NATS request client to the PD router too, so disaggregated
        # (prefill/decode) dispatch uses the per-instance NATS transport when
        # configured (it falls back to HTTP only when nats_client is None).
        self._disagg = DisaggRouter(self.pool, self.policy, nats_client=self.nats_client)

    async def aclose(self) -> None:
        await self._mixed.aclose()
        await self._disagg.aclose()

    async def dispatch(
        self,
        body: dict,
        *,
        stream: bool,
        path: str = "/v1/chat/completions",
    ) -> Response:
        model = body.get("model")
        has_p = self.pool.list_active(model=model, mode=DisaggMode.PREFILL)
        has_d = self.pool.list_active(model=model, mode=DisaggMode.DECODE)
        if has_p and has_d:
            return await self._disagg.dispatch(body, stream=stream, path=path)
        return await self._mixed.dispatch(body, stream=stream, path=path)
