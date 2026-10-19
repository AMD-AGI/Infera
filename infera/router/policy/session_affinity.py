###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Session affinity layered on top of any routing policy.

Mostly the Rust router's ``--session-affinity``: a request with an
``x-session-id`` header (stashed on the body by the server as
``_infera_session_headers``) goes to the target its session used last, per
role. The wrapped policy picks the target for a new session, and still picks
a pinned one (over just that target), so block hashing and load accounting
are unchanged.

A failed request drops the bindings that still point at the targets it used
(the router reports it through ``on_request_failed``), so the session's next
request is picked afresh.

Unlike Rust, a binding is matched by route key only (a worker that re-registers
under the same id keeps its sessions), a failed PD request drops both roles'
bindings rather than only the failed leg's, and idle time counts from the last
routed request rather than the last finished one.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

from infera.common.worker_pool import WorkerInfo
from infera.router.policy.base import Policy
from infera.router.policy.target import RouteTarget

if TYPE_CHECKING:
    from infera.router.kv_event.client import KvEventClient

logger = logging.getLogger(__name__)

SESSION_AFFINITY_MODES = ("off", "prefill", "both")
MAX_SESSIONS = 65536  # bindings per process, as in the Rust router


class SessionAffinityPolicy(Policy):
    """Pins sessions on top of ``inner``.

    ``mode="prefill"`` pins only prefill picks; ``"both"`` also pins decode
    and mixed picks, each role independently. A binding whose target is no
    longer a candidate is re-picked; one idle for ``ttl_seconds`` is dropped.
    At ``MAX_SESSIONS`` new sessions are routed without a binding.
    """

    def __init__(
        self,
        inner: Policy,
        *,
        mode: str,
        ttl_seconds: int = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if mode not in ("prefill", "both"):
            raise ValueError(f"session affinity mode must be prefill or both, got {mode!r}")
        if not 1 <= ttl_seconds <= 86400:
            raise ValueError(f"session affinity TTL must be 1..86400 seconds, got {ttl_seconds}")
        self.inner = inner
        self._roles = {"prefill"} if mode == "prefill" else {"prefill", "decode", "mixed"}
        self._ttl = ttl_seconds
        self._clock = clock
        # (model, session, role) -> (route_key, last_used), least recently used first.
        self._bindings: OrderedDict[tuple[str, str, str], tuple[str, float]] = OrderedDict()
        logger.info("session affinity mode=%s ttl=%ds", mode, ttl_seconds)

    def pick_targets(
        self,
        targets: list[RouteTarget],
        request: dict,
        *,
        role_hint: str | None = None,
    ) -> tuple[RouteTarget, list[int]]:
        key = self._key(request, role_hint)
        if key is None:
            return self.inner.pick_targets(targets, request, role_hint=role_hint)

        now = self._clock()
        self._expire(now)
        bound = self._bindings.get(key)
        pinned = [t for t in targets if bound and t.route_key == bound[0]]
        if pinned:
            reason = "hit"
        elif bound:
            reason = "invalid"  # bound target is gone or filtered out
        elif len(self._bindings) < MAX_SESSIONS:
            reason = "new"
        else:
            reason = "full"
        target, blocks = self.inner.pick_targets(pinned or targets, request, role_hint=role_hint)
        if reason != "full":
            self._bindings[key] = (target.route_key, now)
            self._bindings.move_to_end(key)
        logger.info(
            "session affinity pick role=%s reason=%s picked=%s", key[2], reason, target.route_key
        )
        return target, blocks

    def on_request_failed(
        self, route_key: str, request: dict, *, role_hint: str | None = None
    ) -> None:
        # Only if still bound there: a later request may already have moved it.
        key = self._key(request, role_hint)
        if key is not None and self._bindings.get(key, ("",))[0] == route_key:
            del self._bindings[key]
            logger.info("session affinity drop role=%s target=%s", key[2], route_key)

    def _key(self, request: dict, role_hint: str | None) -> tuple[str, str, str] | None:
        """The binding key, or None when this request is not pinned."""
        role = role_hint or "mixed"
        session = (request.get("_infera_session_headers") or {}).get("x-session-id")
        if not session or role not in self._roles:
            return None
        return str(request.get("model") or ""), session, role

    def _expire(self, now: float) -> None:
        while self._bindings:
            _, last_used = next(iter(self._bindings.values()))
            if now - last_used < self._ttl:
                break
            self._bindings.popitem(last=False)

    # Everything else is the wrapped policy's.
    def on_worker_added(self, worker: WorkerInfo) -> None:
        self.inner.on_worker_added(worker)

    def on_worker_removed(self, worker_id: str) -> None:
        self.inner.on_worker_removed(worker_id)

    def on_request_started(self, route_key: str, blocks: list[int] | None = None) -> None:
        self.inner.on_request_started(route_key, blocks)

    def on_request_finished(self, route_key: str, blocks: list[int] | None = None) -> None:
        self.inner.on_request_finished(route_key, blocks)

    async def aclose(self) -> None:
        await self.inner.aclose()

    @property
    def kv_client(self) -> KvEventClient | None:
        return self.inner.kv_client

    def count_input_tokens(self, request: dict) -> int | None:
        return self.inner.count_input_tokens(request)
