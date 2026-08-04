###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Circuit breaker: a failing worker must stop being re-picked.

The bug this guards against is not "failover is broken" -- failover works. It is
that failover's memory is per-request, so the *next* request scores a wedged
worker as if nothing happened. Every test here is written against that: the
assertions are about what happens on the second and third request, not the
first.

Time is injected rather than slept, so the cooldown behaviour is tested at full
speed and deterministically.
"""

from __future__ import annotations

import pytest

from infera.router.breaker import BreakerState, CircuitBreaker, is_worker_fault


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class W:
    """Minimal stand-in for WorkerInfo -- the breaker only reads worker_id."""

    def __init__(self, wid: str) -> None:
        self.worker_id = wid


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def cb(clock):
    return CircuitBreaker(failure_threshold=3, cooldown=5.0, max_cooldown=20.0, now=clock)


def test_unknown_worker_is_allowed(cb):
    """A worker never seen must route; the breaker is not an allowlist."""
    assert cb.allows("w1") is True
    assert cb.state_of("w1") is BreakerState.CLOSED


def test_failures_below_threshold_do_not_open(cb):
    for _ in range(2):
        cb.record_failure("w1")
    assert cb.allows("w1") is True
    assert cb.state_of("w1") is BreakerState.CLOSED


def test_opens_at_threshold_and_excludes(cb):
    """The actual bug: after N failures the worker must stop being offered."""
    for _ in range(3):
        cb.record_failure("w1")
    assert cb.state_of("w1") is BreakerState.OPEN
    assert cb.allows("w1") is False


def test_success_resets_the_count(cb):
    """Intermittent failures must not accumulate into a trip."""
    cb.record_failure("w1")
    cb.record_failure("w1")
    cb.record_success("w1")
    cb.record_failure("w1")
    cb.record_failure("w1")
    assert cb.state_of("w1") is BreakerState.CLOSED
    assert cb.allows("w1") is True


def test_half_open_admits_exactly_one_probe(cb, clock):
    for _ in range(3):
        cb.record_failure("w1")
    clock.advance(5.1)
    assert cb.allows("w1") is True, "cooldown elapsed -> one probe admitted"
    assert cb.state_of("w1") is BreakerState.HALF_OPEN
    assert cb.allows("w1") is False, "a second concurrent request must not also probe"


def test_successful_probe_closes(cb, clock):
    for _ in range(3):
        cb.record_failure("w1")
    clock.advance(5.1)
    cb.allows("w1")
    cb.record_success("w1")
    assert cb.state_of("w1") is BreakerState.CLOSED
    assert cb.allows("w1") is True


def test_failed_probe_reopens_with_doubled_cooldown(cb, clock):
    """A wedged worker does not recover on the first retry. A fixed cooldown
    would probe it forever at a constant rate; this asserts the backoff."""
    for _ in range(3):
        cb.record_failure("w1")
    clock.advance(5.1)
    cb.allows("w1")
    cb.record_failure("w1")  # probe fails -> reopen, cooldown 5 -> 10
    assert cb.state_of("w1") is BreakerState.OPEN

    clock.advance(5.1)  # old cooldown would have elapsed
    assert cb.allows("w1") is False, "backoff must have doubled"
    clock.advance(5.0)  # now past 10s
    assert cb.allows("w1") is True


def test_cooldown_is_capped(cb, clock):
    for _ in range(3):
        cb.record_failure("w1")
    for _ in range(6):  # drive the doubling past max_cooldown
        clock.advance(1000.0)
        cb.allows("w1")
        cb.record_failure("w1")
    clock.advance(20.1)  # max_cooldown = 20
    assert cb.allows("w1") is True, "cooldown must not grow without bound"


def test_filter_drops_open_workers(cb):
    ws = [W("good"), W("bad")]
    for _ in range(3):
        cb.record_failure("bad")
    assert [w.worker_id for w in cb.filter(ws)] == ["good"]


def test_filter_returns_all_when_every_worker_is_open(cb):
    """Refusing to route would turn a partial outage into a total one. A
    request served by a probably-bad worker beats a guaranteed 503."""
    ws = [W("a"), W("b")]
    for wid in ("a", "b"):
        for _ in range(3):
            cb.record_failure(wid)
    assert {w.worker_id for w in cb.filter(ws)} == {"a", "b"}


def test_filter_of_empty_is_empty(cb):
    assert cb.filter([]) == []


def test_workers_are_independent(cb):
    for _ in range(3):
        cb.record_failure("bad")
    assert cb.allows("good") is True
    assert cb.state_of("good") is BreakerState.CLOSED


def test_success_on_unknown_worker_is_harmless(cb):
    cb.record_success("never-seen")
    assert cb.allows("never-seen") is True


def test_snapshot_reports_trips(cb, clock):
    for _ in range(3):
        cb.record_failure("w1")
    clock.advance(5.1)
    cb.allows("w1")
    cb.record_failure("w1")
    snap = cb.snapshot()["w1"]
    assert snap["state"] == "open"
    assert snap["trips"] == 2, "initial trip plus the failed probe"


@pytest.mark.parametrize("status", [500, 502, 503, 504, 0])
def test_server_errors_are_worker_faults(status):
    assert is_worker_fault(status) is True


@pytest.mark.parametrize("status", [400, 404, 422, 429])
def test_client_errors_are_not_worker_faults(status):
    """A malformed request 400s on every worker it reaches. Counting that as a
    health signal would trip the breaker across an entirely healthy fleet.
    429 is excluded separately: it means "full now", which load accounting
    already routes around, and a doubling cooldown is far too heavy for it."""
    assert is_worker_fault(status) is False


def test_a_bad_client_cannot_trip_the_fleet(cb):
    """Ten malformed requests against three healthy workers must leave all
    three closed."""
    ws = [W("a"), W("b"), W("c")]
    for _ in range(10):
        for w in cb.filter(ws):
            if is_worker_fault(400):
                cb.record_failure(w.worker_id)
    assert all(cb.state_of(w.worker_id) is BreakerState.CLOSED for w in ws)


def test_the_regression_this_exists_for(cb):
    """End to end in breaker terms: a worker that fails every dispatch stops
    being selected, instead of being re-picked on every subsequent request.

    Before this class existed, `tried` was per-request, so the loop below would
    have offered `bad` on all ten requests.
    """
    ws = [W("good"), W("bad")]
    offered_bad = 0
    for _ in range(10):
        candidates = cb.filter(ws)
        if any(w.worker_id == "bad" for w in candidates):
            offered_bad += 1
            cb.record_failure("bad")
        cb.record_success("good")
    assert offered_bad == 3, f"bad worker offered {offered_bad} times, expected 3 (the threshold)"


def test_threshold_zero_disables_it():
    """The documented off switch. Without this, threshold=0 would satisfy
    `failures >= threshold` on the very first failure and trip immediately --
    the exact opposite of what --breaker-failure-threshold=0 promises."""
    off = CircuitBreaker(failure_threshold=0)
    for _ in range(20):
        off.record_failure("w1")
    assert off.allows("w1") is True
    assert off.state_of("w1") is BreakerState.CLOSED
    ws = [W("a"), W("b")]
    assert len(off.filter(ws)) == 2
