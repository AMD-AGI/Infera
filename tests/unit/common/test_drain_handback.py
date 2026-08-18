###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Handing resumable generations back when the drain window runs out.

A generation that outlives the drain used to be cancelled, which the client read
as a failure. Now the ones the router can resume are handed back to it instead
and finish on another worker.

This happens *after* the wait, never in place of it. Moving a generation costs
the next worker a re-read of everything produced so far, and a request that
would have finished on its own within the window should simply be allowed to.

The load-bearing constraint is the other half: a request the router *cannot*
resume must still be left alone. Cutting one early would turn a shutdown the
client never noticed into an error it did.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from infera.common.nats_request import (
    DRAINING_NOTICE,
    HDR_TYPE,
    TYPE_ERROR,
    NatsRequestServer,
)


class _Msg:
    def __init__(self, payload: dict, inbox: str):
        self.data = json.dumps(payload).encode()
        self.reply = inbox
        self.headers = None

    async def ack(self):
        pass


class _Conn:
    """Records replies, so a test can see what the router would have received."""

    def __init__(self):
        self.replies: list[tuple[str, str, bytes]] = []

    async def publish(self, subject, data=b"", headers=None):
        self.replies.append((subject, (headers or {}).get(HDR_TYPE), data))

    async def drain(self):
        pass

    def new_inbox(self):
        return "_INBOX.test"

    def types_for(self, inbox: str) -> list[str]:
        return [t for (subj, t, _d) in self.replies if subj == inbox]

    def payloads_for(self, inbox: str) -> list[bytes]:
        return [d for (subj, _t, d) in self.replies if subj == inbox]


def _server(conn) -> NatsRequestServer:
    srv = NatsRequestServer.__new__(NatsRequestServer)
    srv._nc = conn
    srv._inflight = {}
    srv._migratable = set()
    srv._handed_back = set()
    srv._sub = None
    srv._cancel_sub = None
    srv._js = None
    srv._worker_id = "w1"
    srv._max_duration = 0
    srv._http = None
    return srv


async def _never_ends():
    await asyncio.Event().wait()


def _accept(srv, inbox: str, *, migratable: bool) -> asyncio.Task:
    """Register an in-flight request the way _on_request would."""
    task = asyncio.create_task(_never_ends())
    srv._inflight[inbox] = task
    if json.loads(_Msg({"migratable": migratable}, inbox).data).get("migratable"):
        srv._migratable.add(inbox)
    return task


@pytest.mark.asyncio
async def test_a_resumable_generation_is_returned_at_once():
    conn = _Conn()
    srv = _server(conn)
    task = _accept(srv, "inbox-a", migratable=True)

    await srv._hand_back_migratable()
    await asyncio.sleep(0)

    assert conn.types_for("inbox-a") == [TYPE_ERROR]
    assert conn.payloads_for("inbox-a") == [DRAINING_NOTICE]
    assert task.cancelled() or task.cancelling(), "the engine must stop generating"


@pytest.mark.asyncio
async def test_a_generation_nobody_can_resume_is_left_alone():
    """The whole feature is opt-in from the router's side. Without that promise
    the worker must drain the slow way, or it converts a shutdown the client
    never saw into a visible failure."""
    conn = _Conn()
    srv = _server(conn)
    task = _accept(srv, "inbox-b", migratable=False)

    await srv._hand_back_migratable()
    await asyncio.sleep(0)

    assert conn.replies == [], "nothing may be sent for a request that cannot move"
    assert not task.done()
    task.cancel()


@pytest.mark.asyncio
async def test_only_the_resumable_half_is_returned():
    conn = _Conn()
    srv = _server(conn)
    movable = _accept(srv, "yes", migratable=True)
    staying = _accept(srv, "no", migratable=False)

    await srv._hand_back_migratable()
    await asyncio.sleep(0)

    assert conn.types_for("yes") == [TYPE_ERROR]
    assert conn.types_for("no") == []
    assert movable.cancelled() or movable.cancelling()
    assert not staying.done()
    staying.cancel()


@pytest.mark.asyncio
async def test_the_handover_is_announced_once():
    """The cancellation that follows also reports an error. Two frames for one
    event would have the router read a planned handover as a second failure."""
    conn = _Conn()
    srv = _server(conn)
    _accept(srv, "inbox-c", migratable=True)

    await srv._hand_back_migratable()
    assert "inbox-c" in srv._handed_back, "the cancel path must know to stay quiet"

    # What _proxy does when the cancellation lands.
    if "inbox-c" not in srv._handed_back:
        await srv._reply("inbox-c", TYPE_ERROR, b"request cancelled")
    assert conn.types_for("inbox-c") == [TYPE_ERROR]


@pytest.mark.asyncio
async def test_a_finished_request_is_not_disturbed():
    conn = _Conn()
    srv = _server(conn)
    done = asyncio.create_task(asyncio.sleep(0))
    await done
    srv._inflight["gone"] = done
    srv._migratable.add("gone")

    await srv._hand_back_migratable()
    assert conn.replies == []


@pytest.mark.asyncio
async def test_nothing_in_flight_is_a_no_op():
    conn = _Conn()
    srv = _server(conn)
    await srv._hand_back_migratable()
    assert conn.replies == []


@pytest.mark.asyncio
async def test_the_wait_comes_first_and_the_handover_after():
    """A request that finishes inside the drain window costs nothing. Handing
    it over early would buy the next worker a re-read for no reason."""
    conn = _Conn()
    srv = _server(conn)
    finished = asyncio.create_task(asyncio.sleep(0))
    await finished
    srv._inflight["quick"] = finished
    srv._migratable.add("quick")
    still_going = _accept(srv, "slow", migratable=True)

    await srv.stop(drain=True, drain_timeout=0.01)
    await asyncio.sleep(0)

    assert conn.types_for("quick") == [], "it finished on its own; nothing to hand back"
    assert conn.payloads_for("slow") == [DRAINING_NOTICE]
    assert still_going.cancelled() or still_going.cancelling()


@pytest.mark.asyncio
async def test_a_zero_window_still_hands_over_rather_than_cutting():
    """`--drain-timeout 0` means leave now, not sever what could have lived.
    The handover is a local publish, so it costs nothing to honour that."""
    conn = _Conn()
    srv = _server(conn)
    task = _accept(srv, "inbox-z", migratable=True)

    await srv.stop(drain=True, drain_timeout=0)
    await asyncio.sleep(0)

    assert conn.payloads_for("inbox-z") == [DRAINING_NOTICE]
    assert task.cancelled() or task.cancelling()


@pytest.mark.asyncio
async def test_an_abrupt_stop_hands_nothing_over():
    """Not a drain: this is the emergency path, where the caller has asked for
    everything to stop at once rather than for an orderly exit."""
    conn = _Conn()
    srv = _server(conn)
    task = _accept(srv, "inbox-a", migratable=True)

    await srv.stop()
    await asyncio.sleep(0)

    assert conn.replies == []
    assert task.cancelled() or task.cancelling()
