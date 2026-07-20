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

import pytest

from infera.common.nats_request import NatsRequestServer


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
