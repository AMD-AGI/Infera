###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import logging

from fastapi import Response
from fastapi.responses import JSONResponse

from infera.common.worker_pool import DisaggMode
from infera.router.base import BaseRouter
from infera.router.disagg import DisaggRouter
from infera.router.mixed import MixedRouter

logger = logging.getLogger(__name__)


class AutoRouter(BaseRouter):
    """Per-request router selector.

    Selection policy (PD-preferred with mixed fallback):
      - BOTH prefill and decode workers                      → DisaggRouter
      - Exactly one PD pool, and no mixed workers            → 503 naming the
        empty pool (half a PD deployment cannot serve, and saying "no mixed
        worker" would describe something the operator never deployed)
      - Otherwise (mixed workers present, or nothing at all) → MixedRouter
        (MixedRouter itself returns 503 if no mixed worker is available)

    This supports mixed deployments (some models PD, others mixed) and rolling
    upgrades without a server restart. The two inner routers share the pool
    and policy, so any state held by the policy (e.g. round-robin counters)
    is fleet-wide.

    Future evolution: smarter selection (e.g. selective PD by prompt length /
    prefix-cache hit) belongs in the Policy layer, not here. This router stays
    a dumb dispatcher.
    """

    def __init__(self, *args, migration_limit: int = 0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._mixed = MixedRouter(
            self.pool,
            self.policy,
            nats_client=self.nats_client,
            request_max_retries=self.request_max_retries,
            # Only mixed workers can carry a generation elsewhere; the PD path
            # has a second leg whose state would also have to move.
            migration_limit=migration_limit,
            # One breaker shared by both sub-routers: otherwise each would build
            # its own default and the configured thresholds would never reach
            # them, since AutoRouter is what the server actually constructs.
            breaker=self.breaker,
        )
        # Pass the NATS request client to the PD router too, so disaggregated
        # (prefill/decode) dispatch uses the per-instance NATS transport when
        # configured (it falls back to HTTP only when nats_client is None).
        self._disagg = DisaggRouter(
            self.pool, self.policy, nats_client=self.nats_client, breaker=self.breaker
        )

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
        # Exactly one PD pool populated: the deployment is disaggregated but
        # half of it is gone. Falling through to the mixed router would be
        # correct-but-useless -- there are no mixed workers either, so it
        # answers "no active mixed worker", which sends the reader looking for
        # something they never deployed while a decode (or prefill) pool sits
        # right there. Scaling either side to zero is the usual cause.
        if bool(has_p) != bool(has_d) and not self.pool.list_active(
            model=model, mode=DisaggMode.MIXED
        ):
            present, missing = ("prefill", "decode") if has_p else ("decode", "prefill")
            logger.warning(
                "model=%r has %d %s worker(s) but no %s worker: PD dispatch needs both",
                model,
                len(has_p or has_d),
                present,
                missing,
            )
            return JSONResponse(
                content={
                    "error": (
                        f"model={model!r} has {len(has_p or has_d)} {present} worker(s) "
                        f"but no {missing} worker; PD dispatch requires both pools"
                    )
                },
                status_code=503,
            )
        return await self._mixed.dispatch(body, stream=stream, path=path)
