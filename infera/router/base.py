###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from abc import ABC, abstractmethod

from fastapi import Response

from infera.common.worker_pool import WorkerPool
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
