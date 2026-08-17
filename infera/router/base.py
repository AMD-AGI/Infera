###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import Response

from infera.common.worker_pool import WorkerPool
from infera.router.breaker import CircuitBreaker
from infera.router.policy.base import Policy


class BaseRouter(ABC):
    """Owns the per-request transport protocol.

    Implementations decide how to talk to the chosen worker(s):
      - MixedRouter : single forward
      - DisaggRouter: PD dual-dispatch; per-protocol body shaping +
                      topology (concurrent push or serial pull).
    """

    def __init__(
        self,
        pool: WorkerPool,
        policy: Policy,
        nats_client=None,
        request_max_retries: int = 1,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.pool = pool
        self.policy = policy
        # Optional NatsRequestClient. When set, workers whose request_transport
        # is "nats" are reached over NATS instead of direct HTTP. Selection /
        # policy are unchanged; this only swaps the per-worker send transport.
        self.nats_client = nats_client
        # Bounded failover: how many ALTERNATE workers to try if a dispatch
        # fails BEFORE any response data has been streamed to the client. 0
        # disables retries (single attempt). Mid-stream failures are never
        # retried (output already partially sent).
        self.request_max_retries = max(0, request_max_retries)
        # Per-worker failure memory across requests. Failover alone forgets a
        # bad worker the moment the request ends, so the next one re-picks it.
        # Subclasses that select their own target consult this when filtering
        # candidates; DirectRouter does not select and leaves it unused.
        self.breaker = breaker if breaker is not None else CircuitBreaker()

    @abstractmethod
    async def dispatch(
        self,
        body: dict,
        *,
        stream: bool,
        path: str = "/v1/chat/completions",
    ) -> Response: ...

    async def aclose(self) -> None:
        """Release any resources (HTTP clients, etc). Default no-op."""
        return
