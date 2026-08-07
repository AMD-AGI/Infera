###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""NatsRequestServer.stop() graceful drain: on a rolling upgrade the worker
should let in-flight requests finish (up to drain_timeout) before cancelling,
instead of severing them immediately."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from infera.common.nats_request import REQUEST_STREAM, NatsRequestServer, request_durable


@pytest.mark.asyncio
async def test_drain_waits_for_inflight_then_cancels_leftovers():
    srv = NatsRequestServer("w:1", 30000)
    fast = asyncio.create_task(asyncio.sleep(0.05))  # finishes within the drain window
    slow = asyncio.create_task(asyncio.sleep(30))  # exceeds it -> cancelled
    srv._inflight = {"fast": fast, "slow": slow}

    await srv.stop(drain=True, drain_timeout=0.5)

    assert fast.done() and not fast.cancelled()
    await asyncio.gather(slow, return_exceptions=True)
    assert slow.cancelled()


@pytest.mark.asyncio
async def test_no_drain_cancels_in_flight_immediately():
    srv = NatsRequestServer("w:1", 30000)
    t = asyncio.create_task(asyncio.sleep(30))
    srv._inflight = {"a": t}

    await srv.stop()  # drain=False (default) -> cancel at once

    await asyncio.gather(t, return_exceptions=True)
    assert t.cancelled()


# --- JetStream (admission throttle) path -------------------------------------
#
# Under the throttle the router publishes into a WorkQueue stream and this
# worker pulls from it, so a request can be accepted, queued, and invisible to
# `_inflight` -- which only tracks what has already been delivered here.


class _FakeJs:
    """JetStream stand-in: a scripted num_pending sequence + a call log."""

    def __init__(self, pending: list[int] | None = None, *, fail: bool = False) -> None:
        self._pending = list(pending or [])
        self._fail = fail
        self.deleted: list[tuple[str, str]] = []
        self.info_calls = 0

    async def consumer_info(self, stream, consumer, timeout=None):
        self.info_calls += 1
        if self._fail:
            raise RuntimeError("broker hiccup")
        value = self._pending.pop(0) if self._pending else 0
        return SimpleNamespace(num_pending=value, num_ack_pending=0)

    async def delete_consumer(self, stream, consumer):
        self.deleted.append((stream, consumer))
        return True


class _FakeSub:
    """Records when the subject was unsubscribed, in units of backlog polls."""

    def __init__(self, js: _FakeJs) -> None:
        self._js = js
        self.unsubscribed_after_polls: int | None = None

    async def unsubscribe(self):
        self.unsubscribed_after_polls = self._js.info_calls


@pytest.mark.asyncio
async def test_the_door_closes_only_after_the_backlog_clears():
    """`unsubscribe()` discards whatever is left in the stream, so it must not
    run until the backlog has been handed over -- otherwise a request the
    router already accepted is silently dropped and the client waits out the
    full idle timeout for a reply nobody will send.

    Pinned by *when* the unsubscribe happened, not just that the backlog was
    polled: polling and then closing the door anyway would pass a weaker check.
    """
    js = _FakeJs([2, 1, 0])
    sub = _FakeSub(js)
    srv = NatsRequestServer("w:1", 30000)
    srv._js = js
    srv._sub = sub

    await srv.stop(drain=True, drain_timeout=5)

    assert sub.unsubscribed_after_polls is not None, "unsubscribe never ran"
    assert sub.unsubscribed_after_polls >= 3, (
        "unsubscribed while the stream still held work: only "
        f"{sub.unsubscribed_after_polls} poll(s) had happened, backlog clears on the 3rd"
    )


@pytest.mark.asyncio
async def test_a_backlog_that_never_clears_is_bounded_by_the_deadline():
    """A stuck backlog must not hold the process past its drain budget -- the
    kubelet's SIGKILL does not wait, and everything after this still has to
    run."""
    js = _FakeJs([5] * 1000)
    srv = NatsRequestServer("w:1", 30000)
    srv._js = js

    start = asyncio.get_running_loop().time()
    await srv.stop(drain=True, drain_timeout=0.5)
    elapsed = asyncio.get_running_loop().time() - start

    assert elapsed < 3.0, f"stop() overran its 0.5s budget by far: {elapsed:.1f}s"


@pytest.mark.asyncio
async def test_an_unreadable_backlog_does_not_block_shutdown():
    """A stalled rollout is worse than a dropped queued request, and a broker
    that cannot answer looks identical to a genuinely busy one."""
    js = _FakeJs(fail=True)
    srv = NatsRequestServer("w:1", 30000)
    srv._js = js

    await asyncio.wait_for(srv.stop(drain=True, drain_timeout=30), timeout=2.0)

    assert js.info_calls == 1, "should give up after the first failed read"


@pytest.mark.asyncio
async def test_stop_deletes_the_durable_consumer():
    """The durable outlives the subscription by definition, and its name comes
    from worker_id -- which a rebuilt Pod never reuses. Left behind, every
    rollout adds an orphan holding WorkQueue quota nothing will consume."""
    js = _FakeJs([0])
    srv = NatsRequestServer("10.0.0.1:30000", 30000)
    srv._js = js

    await srv.stop(drain=True, drain_timeout=1)

    assert len(js.deleted) == 1, f"expected one delete_consumer call, got {js.deleted}"
    stream, consumer = js.deleted[0]
    assert stream == REQUEST_STREAM
    assert consumer == request_durable("10.0.0.1:30000")


@pytest.mark.asyncio
async def test_core_nats_path_touches_no_jetstream():
    """Without the throttle there is no stream and no durable; the shutdown
    path must not assume otherwise."""
    srv = NatsRequestServer("w:1", 30000)
    assert srv._js is None

    await srv.stop(drain=True, drain_timeout=1)  # must not raise
