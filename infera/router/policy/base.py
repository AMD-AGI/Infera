###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from infera.common.worker_pool import WorkerInfo
from infera.router.policy.target import RouteTarget

if TYPE_CHECKING:
    from infera.router.kv_event.client import KvEventClient


class Policy(ABC):
    """Picks a routing target for a request; lifecycle hooks default to no-op."""

    @abstractmethod
    def pick(
        self,
        candidates: list[WorkerInfo],
        request: dict,
        *,
        role_hint: str | None = None,
    ) -> tuple[RouteTarget, list[int]]:
        """Returns ``(target, blocks)`` where ``target`` is the picked
        ``(worker, dp_rank)`` and ``blocks`` is the chained-hash list of the
        request's KV blocks under the picked worker's ``kv_block_size``. The
        router echoes ``blocks`` back through ``on_request_started`` /
        ``on_request_finished`` (keyed by ``target.route_key``) so cost-aware
        policies can refcount unique in-flight blocks per target without
        re-tokenising. Stateless policies return ``[]``.

        ``role_hint`` is set by the disagg-bootstrap router to ``"prefill"``
        or ``"decode"`` so cost-aware policies can apply per-role tuning
        (e.g. weight cache locality higher for prefill, load balance higher
        for decode). Mixed routers leave it ``None``.
        """

    # Registry fires these on PUT / DELETE events from etcd.
    def on_worker_added(self, worker: WorkerInfo) -> None:  # noqa: B027
        pass

    def on_worker_removed(self, worker_id: str) -> None:  # noqa: B027
        pass

    # Router fires these around dispatch (for in-flight tracking, metrics, ...).
    # The key is ``RouteTarget.route_key`` (== worker_id for single-rank targets).
    def on_request_started(  # noqa: B027
        self, route_key: str, blocks: list[int] | None = None
    ) -> None:
        pass

    def on_request_finished(  # noqa: B027
        self, route_key: str, blocks: list[int] | None = None
    ) -> None:
        pass

    async def aclose(self) -> None:  # noqa: B027
        pass

    @property
    def kv_client(self) -> KvEventClient | None:
        return None
