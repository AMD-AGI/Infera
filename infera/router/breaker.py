###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Per-worker circuit breaker for routers that select their own target.

Failover alone is not enough. It retries a failed dispatch on another worker,
but the memory of that failure lives in a per-request ``tried`` set that is
discarded when the request returns -- so the next request scores the same broken
worker as if nothing happened, picks it again if cache locality says so, and
pays the failover cost again. A worker that is healthy to the platform and
broken for inference therefore taxes *every* request, indefinitely.

That worker is not hypothetical: it accepts the connection, answers ``/health``,
stays ``ACTIVE`` in discovery, and fails before the first byte. Kubernetes
cannot see it and neither can discovery. The router is the only component that
knows, and without this it forgets immediately.

Scope, deliberately narrow:

* **Only pre-first-byte failures trip it.** A failure after bytes have been
  streamed is already non-retryable by design -- the worker demonstrably served
  part of the request, and treating that as a health signal would open the
  breaker on ordinary client disconnects.
* **It never touches ``WorkerStatus``.** That field is owned by discovery; this
  is the router's private opinion, applied when filtering candidates.
* **Only routers that select use it.** ``direct.py`` has no failover because the
  gateway owns selection there, so a breaker would be wrong.

States are the usual three. ``closed`` routes normally. After
``failure_threshold`` consecutive failures the breaker goes ``open`` and the
worker is excluded for ``cooldown`` seconds. It then becomes ``half_open`` and
admits one probe at a time: success closes it and clears the count, failure
reopens it with the cooldown doubled, up to ``max_cooldown``. Backing off
matters because the common cause -- a worker wedged on a bad KV handoff -- does
not resolve on the first retry, and a fixed cooldown turns into a probe every
``cooldown`` seconds forever.

"One at a time" is bounded by ``probe_timeout`` rather than by waiting for an
outcome, because the outcome may never arrive: the slot is claimed while
filtering candidates, and only the one the policy dispatches to reports back.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from infera.server import metrics

logger = logging.getLogger(__name__)

_STATE_VALUE = {"closed": 0, "half_open": 1, "open": 2}


def _observe(worker_id: str, state) -> None:
    """Mirror a state change to Prometheus. Never allowed to fail a request --
    an unregistered collector or a duplicate registry must not take out the
    data plane."""
    try:
        metrics.worker_breaker_state.labels(worker_id=worker_id).set(_STATE_VALUE[state.value])
    except Exception:  # pragma: no cover
        pass


def is_worker_fault(status: int) -> bool:
    """True if an HTTP status is evidence about the *worker*, not the request.

    Failover retries on any pre-first-byte error, including 4xx -- that is
    correct, since a 400 costs nothing to re-ask. Feeding 4xx to the breaker is
    not: a malformed request returns 400 from every worker it touches, so the
    breaker would trip the entire healthy fleet on one bad client. 429 is
    excluded for a different reason -- it means "full right now", which the
    policy's load accounting already routes around, and a 5s cooldown with
    doubling is far too heavy a response to transient backpressure.
    """
    return status >= 500 or status == 0


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Entry:
    consecutive_failures: int = 0
    state: BreakerState = BreakerState.CLOSED
    # Wall time after which an open breaker becomes half-open.
    opens_until: float = 0.0
    # Cooldown applied on the *next* trip; doubles each time a probe fails.
    next_cooldown: float = 0.0
    # When the outstanding half-open probe was admitted, so only one is in
    # flight at a time. None means the slot is free.
    probe_started_at: float | None = None
    trips: int = 0


@dataclass
class CircuitBreaker:
    """Tracks per-worker health as seen by dispatch outcomes.

    Not thread-safe by design: routers drive it from a single asyncio loop, and
    a lock here would sit on the hot path of every request for no benefit.
    """

    failure_threshold: int = 3
    cooldown: float = 5.0
    max_cooldown: float = 60.0
    #: How long a claimed probe slot is honoured before it is reclaimed.
    #:
    #: Claiming and releasing the slot are not paired: ``filter`` claims one for
    #: every candidate it lets through, and the policy then dispatches exactly
    #: one of them, so the rest are never told how they did. A 4xx, a client
    #: disconnect or a request that never returns leaves the slot held too.
    #: Bounding the claim keeps any of those from wedging a healthy worker out
    #: of rotation permanently. The cost of reclaiming too early is one extra
    #: probe; the cost of never reclaiming is a worker lost until restart.
    probe_timeout: float = 60.0
    #: Injectable clock, so tests do not sleep.
    now: object = field(default=time.monotonic)
    _entries: dict[str, _Entry] = field(default_factory=dict, init=False)

    # --- queries ------------------------------------------------------------

    def _entry(self, worker_id: str) -> _Entry:
        e = self._entries.get(worker_id)
        if e is None:
            e = _Entry(next_cooldown=self.cooldown)
            self._entries[worker_id] = e
        return e

    @property
    def enabled(self) -> bool:
        """A threshold of 0 or less turns the breaker off entirely, so an
        operator can fall back to plain failover without a code change."""
        return self.failure_threshold > 0

    def allows(self, worker_id: str) -> bool:
        """True if this worker may be dispatched to right now.

        Transitions open -> half_open as a side effect when the cooldown has
        elapsed, because the alternative is a separate timer whose only job is
        to flip a flag that this function already has to check.
        """
        if not self.enabled:
            return True
        e = self._entries.get(worker_id)
        if e is None or e.state is BreakerState.CLOSED:
            return True
        now = self.now()
        if e.state is BreakerState.OPEN:
            if now < e.opens_until:
                return False
            e.state = BreakerState.HALF_OPEN
            e.probe_started_at = None
            _observe(worker_id, e.state)
            logger.info("breaker: worker %s half-open, admitting one probe", worker_id)
        # half_open: one probe at a time, and only for as long as a probe could
        # plausibly still be running -- see probe_timeout for why the claim has
        # to expire rather than wait for an outcome that may never come.
        if e.probe_started_at is not None:
            if now - e.probe_started_at < self.probe_timeout:
                return False
            logger.info(
                "breaker: worker %s probe slot unclaimed after %.0fs; admitting another",
                worker_id,
                self.probe_timeout,
            )
        e.probe_started_at = now
        return True

    def filter(self, workers):
        """Drop workers whose breaker is open. Returns a list.

        If every candidate is open, returns them all rather than nothing: a
        request served by a probably-bad worker beats a 503 when there is no
        alternative, and refusing to route would turn a partial outage into a
        total one.
        """
        allowed = [w for w in workers if self.allows(self._id_of(w))]
        if allowed:
            return allowed
        if workers:
            logger.warning(
                "breaker: all %d candidate(s) open; routing anyway rather than failing",
                len(workers),
            )
        return list(workers)

    @staticmethod
    def _id_of(w) -> str:
        # Accepts a WorkerInfo or anything exposing .worker_id.
        return getattr(w, "worker_id", None) or str(w)

    def state_of(self, worker_id: str) -> BreakerState:
        e = self._entries.get(worker_id)
        return e.state if e else BreakerState.CLOSED

    # --- outcomes -----------------------------------------------------------

    def record_success(self, worker_id: str) -> None:
        e = self._entries.get(worker_id)
        if e is None:
            return
        if e.state is not BreakerState.CLOSED:
            logger.info("breaker: worker %s recovered, closing", worker_id)
        e.consecutive_failures = 0
        e.state = BreakerState.CLOSED
        e.probe_started_at = None
        e.next_cooldown = self.cooldown
        _observe(worker_id, e.state)

    def record_neutral(self, worker_id: str) -> None:
        """Release the probe slot without scoring the worker either way.

        For an outcome that says nothing about worker health -- a 4xx, which
        every worker would answer identically, or a 429, which is backpressure
        the policy already routes around. Counting it as recovery is as wrong
        as counting it as failure: it would reset the failure count and close
        an open breaker, so a worker alternating 500s and 400s would never
        accumulate the consecutive failures needed to trip. But the slot such a
        request consumed must still come back, or one bad client can wedge a
        recovering worker out of rotation.
        """
        e = self._entries.get(worker_id)
        if e is None:
            return
        e.probe_started_at = None

    def forget(self, worker_id: str) -> None:
        """Drop everything remembered about a worker that has left the fleet.

        Worker ids are addresses and a rebuilt Pod never reuses one, so without
        this every rollout strands another entry -- and another pair of
        Prometheus series, since both are labelled by worker id.
        """
        if self._entries.pop(worker_id, None) is None:
            return
        for collector in (metrics.worker_breaker_state, metrics.worker_breaker_trips_total):
            try:
                collector.remove(worker_id)
            except Exception:  # noqa: BLE001 - never registered, or already gone
                pass

    def record_failure(self, worker_id: str) -> None:
        """Record a pre-first-byte dispatch failure."""
        if not self.enabled:
            return
        e = self._entry(worker_id)
        e.consecutive_failures += 1
        was_probe = e.state is BreakerState.HALF_OPEN
        e.probe_started_at = None

        if was_probe:
            # A failed probe reopens immediately and backs off further, without
            # waiting for the threshold again -- we already know it is bad.
            e.next_cooldown = min(e.next_cooldown * 2, self.max_cooldown)
            self._open(worker_id, e)
            return
        if e.consecutive_failures >= self.failure_threshold:
            self._open(worker_id, e)

    def _open(self, worker_id: str, e: _Entry) -> None:
        # A trip is an edge into exclusion, not every failure that lands while
        # the worker is already excluded. A failed probe counts: it is a fresh
        # verdict on a worker that was given another chance, and the doubling
        # cooldown bounds how often one can happen. A failure while already
        # open does not -- those arrive at the request rate, via the all-open
        # fallback, and counting them turns the metric into a request counter
        # that drowns out real trips and prints the warning on every request.
        newly_tripped = e.state is not BreakerState.OPEN
        e.state = BreakerState.OPEN
        e.opens_until = self.now() + e.next_cooldown
        _observe(worker_id, e.state)
        if not newly_tripped:
            return
        e.trips += 1
        try:
            metrics.worker_breaker_trips_total.labels(worker_id=worker_id).inc()
        except Exception:  # pragma: no cover - metrics must never break routing
            pass
        logger.warning(
            "breaker: worker %s open for %.1fs after %d consecutive failure(s)",
            worker_id,
            e.next_cooldown,
            e.consecutive_failures,
        )

    # --- introspection for metrics / tests -----------------------------------

    def snapshot(self) -> dict[str, dict]:
        return {
            wid: {
                "state": e.state.value,
                "consecutive_failures": e.consecutive_failures,
                "trips": e.trips,
            }
            for wid, e in self._entries.items()
        }
