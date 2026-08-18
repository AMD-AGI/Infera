###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""SLA signals on the PD-disaggregated path.

The disagg router hands its token stream to one of three generators depending on
topology and transport, and each of them has to close the observer itself. A
missed ``close()`` in any one of them silently starves the planner of TTFT and
ITL for that deployment shape, so the streaming path is covered here as well as
the unary one.

These use SGLang's bootstrap protocol over NATS: it is the concurrent topology,
so both legs are driven through the fake NATS client and no test touches a real
socket.
"""

from __future__ import annotations

import json

import pytest

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import DisaggMode, EngineType, WorkerInfo
from infera.router.disagg import DisaggRouter
from infera.router.policy.target import RouteTarget
from infera.server import metrics

from .test_sla_metrics import histogram_totals, sse


def pd_worker(wid: str, mode: DisaggMode, kv_block_size: int | None = None) -> WorkerInfo:
    return WorkerInfo(
        worker_id=wid,
        url=f"http://{wid}",
        model_name="m",
        engine=EngineType.SGLANG,
        disagg_mode=mode,
        disagg_meta={
            "protocol": "sglang-bootstrap",
            "params": {"bootstrap_addr": f"{wid.rsplit(':', 1)[0]}:8998"},
        },
        request_transport="nats",
        kv_block_size=kv_block_size,
    )


class FakePolicy:
    def __init__(self, blocks: list[int] | None = None) -> None:
        self._blocks = blocks or []

    def pick(self, candidates, body, role_hint=None):
        return RouteTarget(candidates[0]), list(self._blocks)

    def on_request_started(self, route_key, blocks):
        pass

    def on_request_finished(self, route_key, blocks):
        pass


class FakePool:
    def __init__(self, prefill, decode):
        self._prefill = prefill
        self._decode = decode

    def list_active(self, model=None, mode=None):
        if mode is DisaggMode.PREFILL:
            return list(self._prefill)
        if mode is DisaggMode.DECODE:
            return list(self._decode)
        return list(self._prefill) + list(self._decode)


class FakeNats:
    """Replies to the prefill leg tersely and replays a script for decode."""

    def __init__(self, decode_id: str, decode_events):
        self._decode_id = decode_id
        self._decode_events = decode_events

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        if worker_id == self._decode_id:
            for event in self._decode_events:
                yield event
            return
        yield (TYPE_DATA, None, b"{}")
        yield (TYPE_DONE, 200, b"")


def router_for(decode_events, *, blocks: list[int] | None = None) -> DisaggRouter:
    p = pd_worker("10.0.0.1:30001", DisaggMode.PREFILL, kv_block_size=64)
    d = pd_worker("10.0.0.2:30002", DisaggMode.DECODE, kv_block_size=64)
    return DisaggRouter(
        FakePool([p], [d]),
        FakePolicy(blocks=blocks),
        nats_client=FakeNats(d.worker_id, decode_events),
    )


async def drain(response) -> bytes:
    out = b""
    async for chunk in response.body_iterator:
        out += chunk if isinstance(chunk, bytes) else chunk.encode()
    return out


class TestDisaggStreaming:
    async def test_tokens_are_counted_from_the_decode_leg(self):
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0}, {"i": 1}, {"i": 2}, {"i": 3})]
        events.append((TYPE_DONE, 200, b""))
        router = router_for(events)

        before = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        await drain(response)
        after = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        assert after[1] == before[1] + 1
        assert after[0] - before[0] == pytest.approx(4.0)
        await router.aclose()

    async def test_ttft_is_recorded_once_the_stream_drains(self):
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0}, {"i": 1})]
        events.append((TYPE_DONE, 200, b""))
        router = router_for(events)

        before = histogram_totals(metrics.time_to_first_token_seconds, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        assert histogram_totals(metrics.time_to_first_token_seconds, router="disagg") == before
        await drain(response)
        after = histogram_totals(metrics.time_to_first_token_seconds, router="disagg")
        assert after[1] == before[1] + 1
        await router.aclose()

    async def test_isl_comes_from_the_prefill_leg_block_count(self):
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0})]
        events.append((TYPE_DONE, 200, b""))
        router = router_for(events, blocks=[1, 2, 3])

        before = histogram_totals(metrics.input_sequence_tokens, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        await drain(response)
        after = histogram_totals(metrics.input_sequence_tokens, router="disagg")
        assert after[0] - before[0] == pytest.approx(192.0)
        await router.aclose()

    async def test_itl_is_recorded_when_more_than_one_token_is_generated(self):
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0}, {"i": 1}, {"i": 2})]
        events.append((TYPE_DONE, 200, b""))
        router = router_for(events)

        before = histogram_totals(metrics.inter_token_latency_seconds, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        await drain(response)
        after = histogram_totals(metrics.inter_token_latency_seconds, router="disagg")
        assert after[1] == before[1] + 1
        await router.aclose()

    async def test_a_single_token_reply_reports_no_itl(self):
        # ITL needs a gap between two tokens; one token gives no interval to
        # measure, and recording a zero would drag the fleet average down.
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0})]
        events.append((TYPE_DONE, 200, b""))
        router = router_for(events)

        before = histogram_totals(metrics.inter_token_latency_seconds, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        await drain(response)
        after = histogram_totals(metrics.inter_token_latency_seconds, router="disagg")
        assert after == before
        await router.aclose()


class TestDisaggUnary:
    async def test_usage_drives_isl_and_osl(self):
        body = json.dumps({"usage": {"prompt_tokens": 900, "completion_tokens": 150}}).encode()
        router = router_for([(TYPE_DATA, None, body), (TYPE_DONE, 200, b"")])

        isl_before = histogram_totals(metrics.input_sequence_tokens, router="disagg")
        osl_before = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=False)
        assert response.status_code == 200

        isl_after = histogram_totals(metrics.input_sequence_tokens, router="disagg")
        osl_after = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        assert isl_after[0] - isl_before[0] == pytest.approx(900.0)
        assert osl_after[0] - osl_before[0] == pytest.approx(150.0)
        await router.aclose()

    async def test_missing_pd_workers_records_nothing(self):
        router = DisaggRouter(FakePool([], []), FakePolicy())
        before = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        assert response.status_code == 503
        assert histogram_totals(metrics.output_sequence_tokens, router="disagg") == before
        await router.aclose()


class TestStreamFailureIsNotCountedAsTraffic:
    """A leg that dies after hand-off must not be recorded as a served request.

    The streaming routers commit ``outcome="ok"`` when they return the
    StreamingResponse, because that is the point past which the client can no
    longer be failed over. If the decode leg then turns out to be unreachable,
    the observer would close as a success -- inflating the request count the
    planner divides by, and adding a one-frame OSL.

    Observed on real hardware: a decode outage produced 201 `decode_unreachable`
    failures while the request counter showed 100% ok, and the planner sized the
    prefill pool from that window.
    """

    async def test_error_frame_disowns_the_request(self):
        # The decode leg errors before sending a token: the generator emits one
        # SSE error frame and returns.
        router = router_for([(TYPE_ERROR, 502, b"decode exploded")])

        req_before = histogram_totals(
            metrics.request_duration_seconds, router="disagg", outcome="ok"
        )
        isl_before = histogram_totals(metrics.input_sequence_tokens, router="disagg")
        osl_before = histogram_totals(metrics.output_sequence_tokens, router="disagg")

        response = await router.dispatch({"model": "m"}, stream=True)
        body = await drain(response)
        assert b"error" in body, "the client must still be told the stream failed"

        assert histogram_totals(metrics.input_sequence_tokens, router="disagg") == isl_before
        assert histogram_totals(metrics.output_sequence_tokens, router="disagg") == osl_before
        # The pre-existing duration histogram is observed at context exit, before
        # the generator runs, so it still counts this one. That is why the SLA
        # metrics carry their own gate rather than trusting that counter.
        assert (
            histogram_totals(metrics.request_duration_seconds, router="disagg", outcome="ok")[1]
            == req_before[1] + 1
        )
        await router.aclose()

    async def test_a_served_stream_is_still_recorded(self):
        # The guard must not throw away good traffic: same shape, no error frame.
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0}, {"i": 1})]
        events.append((TYPE_DONE, 200, b""))
        router = router_for(events)

        before = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        response = await router.dispatch({"model": "m"}, stream=True)
        await drain(response)
        after = histogram_totals(metrics.output_sequence_tokens, router="disagg")
        assert after[1] == before[1] + 1
        assert after[0] - before[0] == pytest.approx(2.0)
        await router.aclose()
