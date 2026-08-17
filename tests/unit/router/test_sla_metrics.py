###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The SLA signals the router feeds to the planner.

Two things are easy to get wrong here and both are checked below. A streaming
reply outlives the ``track_request`` block -- ``dispatch`` returns a
``StreamingResponse`` and the generator runs afterwards -- so the observer has to
be closed by whoever drains the stream, not by the context manager. And the
histograms are only meaningful for successful requests: a 5xx has no latency
worth averaging.
"""

from __future__ import annotations

import json

import pytest
from prometheus_client import Histogram

from infera.common.nats_request import TYPE_DATA, TYPE_DONE, TYPE_ERROR
from infera.common.worker_pool import EngineType, WorkerInfo
from infera.router.mixed import MixedRouter
from infera.router.policy.target import RouteTarget
from infera.server import metrics


def histogram_totals(histogram: Histogram, **labels) -> tuple[float, float]:
    """Return ``(sum, count)`` for one label set of a histogram."""
    child = histogram.labels(**labels)
    samples = {s.name: s.value for s in child.collect()[0].samples}
    name = histogram._name
    return samples.get(f"{name}_sum", 0.0), samples.get(f"{name}_count", 0.0)


def sse(*payloads: dict) -> list[bytes]:
    """Render OpenAI-style streaming frames, one token per frame."""
    return [f"data: {json.dumps(p)}\n\n".encode() for p in payloads]


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
    def __init__(self, workers):
        self._workers = workers

    def list_active(self, model=None, mode=None):
        return list(self._workers)


class FakeNats:
    """Replays a scripted ``(kind, status, data)`` sequence for one worker."""

    def __init__(self, events):
        self._events = events

    async def admit(self, worker_id):
        return True

    async def stream(self, worker_id, payload):
        for event in self._events:
            yield event


def worker(kv_block_size: int | None = None) -> WorkerInfo:
    return WorkerInfo(
        worker_id="w1",
        url="http://w1",
        model_name="m",
        engine=EngineType.SGLANG,
        request_transport="nats",
        kv_block_size=kv_block_size,
    )


async def drain(response) -> bytes:
    out = b""
    async for chunk in response.body_iterator:
        out += chunk if isinstance(chunk, bytes) else chunk.encode()
    return out


class _Sink:
    """Stands in for a histogram child, recording observations."""

    def __init__(self, recorded: list) -> None:
        self._recorded = recorded

    def observe(self, value):
        self._recorded.append(value)


class TestRequestObserver:
    """The accumulator itself, independent of any router."""

    def test_no_signals_are_emitted_for_a_failed_request(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            metrics.time_to_first_token_seconds, "labels", lambda **kw: _Sink(recorded)
        )
        obs = metrics.RequestObserver("mixed")
        obs["outcome"] = "502"
        obs.first_token()
        obs.close()
        assert recorded == []

    def test_close_is_idempotent(self, monkeypatch):
        recorded = []
        monkeypatch.setattr(
            metrics.time_to_first_token_seconds, "labels", lambda **kw: _Sink(recorded)
        )
        obs = metrics.RequestObserver("mixed")
        obs["outcome"] = "ok"
        obs.first_token()
        obs.close()
        obs.close()
        assert len(recorded) == 1

    def test_counts_one_token_per_sse_frame(self):
        obs = metrics.RequestObserver("mixed")
        for chunk in sse({"i": 0}, {"i": 1}, {"i": 2}):
            obs.observe_stream_chunk(chunk)
        assert obs._frames == 3

    def test_frames_split_across_chunk_boundaries_are_counted_once(self):
        # aiter_raw yields transport-sized chunks, so a frame can straddle them.
        # Double-counting here would inflate OSL and deflate ITL.
        obs = metrics.RequestObserver("mixed")
        payload = b"".join(sse({"i": 0}, {"i": 1}, {"i": 2}))
        for i in range(0, len(payload), 7):
            obs.observe_stream_chunk(payload[i : i + 7])
        assert obs._frames == 3

    def test_the_done_sentinel_is_not_a_token(self):
        obs = metrics.RequestObserver("mixed")
        obs.observe_stream_chunk(b"".join(sse({"i": 0}, {"i": 1})))
        obs.observe_stream_chunk(b"data: [DONE]\n\n")
        assert obs._frames == 2

    def test_a_usage_frame_supplies_exact_counts_instead_of_a_token(self):
        # Clients that ask for stream_options.include_usage get a trailing frame
        # carrying the real totals, which beat the frame-count estimate.
        obs = metrics.RequestObserver("mixed")
        obs.observe_stream_chunk(b"".join(sse({"i": 0}, {"i": 1})))
        obs.observe_stream_chunk(
            b'data: {"usage": {"prompt_tokens": 512, "completion_tokens": 99}}\n\n'
        )
        assert obs._frames == 2
        assert obs._isl == 512
        assert obs._osl == 99

    def test_usage_from_a_unary_reply_is_read(self):
        obs = metrics.RequestObserver("mixed")
        obs.observe_usage({"usage": {"prompt_tokens": 1024, "completion_tokens": 200}})
        assert obs._isl == 1024
        assert obs._osl == 200

    @pytest.mark.parametrize(
        "payload", [None, "text", {}, {"usage": None}, {"usage": {"prompt_tokens": "many"}}]
    )
    def test_a_reply_without_usable_usage_is_ignored(self, payload):
        obs = metrics.RequestObserver("mixed")
        obs.observe_usage(payload)
        assert obs._isl is None
        assert obs._osl is None

    def test_blocks_estimate_isl_when_usage_is_absent(self):
        obs = metrics.RequestObserver("mixed")
        obs.observe_blocks([1, 2, 3, 4], 64)
        assert obs._isl == 256

    def test_exact_usage_wins_over_the_block_estimate(self):
        obs = metrics.RequestObserver("mixed")
        obs.observe_usage({"usage": {"prompt_tokens": 200}})
        obs.observe_blocks([1, 2, 3, 4], 64)
        assert obs._isl == 200

    def test_a_stateless_policy_contributes_no_estimate(self):
        # Round-robin returns no blocks, so there is nothing to estimate from.
        obs = metrics.RequestObserver("mixed")
        obs.observe_blocks([], 64)
        obs.observe_blocks([1, 2], None)
        assert obs._isl is None


class TestMixedRouterStreaming:
    async def test_signals_are_emitted_only_after_the_stream_drains(self):
        # The whole point of deferring close(): if the context manager closed the
        # observer, TTFT would be recorded but OSL and ITL would always be zero.
        w = worker(kv_block_size=64)
        router = MixedRouter(
            FakePool([w]), FakePolicy(blocks=[1, 2, 3, 4]), nats_client=FakeNats([])
        )
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0}, {"i": 1}, {"i": 2})]
        events.append((TYPE_DONE, 200, b""))
        router.nats_client = FakeNats(events)

        before = histogram_totals(metrics.output_sequence_tokens, router="mixed")
        response = await router.dispatch({"model": "m"}, stream=True)
        mid = histogram_totals(metrics.output_sequence_tokens, router="mixed")
        assert mid == before, "nothing should be recorded before the stream is drained"

        assert await drain(response)
        after = histogram_totals(metrics.output_sequence_tokens, router="mixed")
        assert after[1] == before[1] + 1
        assert after[0] == before[0] + 3
        await router.aclose()

    async def test_isl_falls_back_to_the_router_block_count(self):
        # A streaming reply carries no usage, so the KV block list the policy
        # already computed is the only prompt-length signal available.
        w = worker(kv_block_size=64)
        events = [(TYPE_DATA, None, c) for c in sse({"i": 0})]
        events.append((TYPE_DONE, 200, b""))
        router = MixedRouter(
            FakePool([w]), FakePolicy(blocks=[1, 2, 3, 4]), nats_client=FakeNats(events)
        )

        before = histogram_totals(metrics.input_sequence_tokens, router="mixed")
        response = await router.dispatch({"model": "m"}, stream=True)
        await drain(response)
        after = histogram_totals(metrics.input_sequence_tokens, router="mixed")
        assert after[0] - before[0] == pytest.approx(256.0)
        await router.aclose()

    async def test_a_pre_first_byte_failure_records_nothing(self):
        w = worker()
        router = MixedRouter(
            FakePool([w]),
            FakePolicy(),
            nats_client=FakeNats([(TYPE_ERROR, 503, b"busy")]),
            request_max_retries=0,
        )
        before = histogram_totals(metrics.time_to_first_token_seconds, router="mixed")
        response = await router.dispatch({"model": "m"}, stream=True)
        assert response.status_code == 503
        after = histogram_totals(metrics.time_to_first_token_seconds, router="mixed")
        assert after == before
        await router.aclose()


class TestMixedRouterUnary:
    async def test_usage_drives_isl_and_osl(self):
        w = worker()
        body = json.dumps(
            {"choices": [], "usage": {"prompt_tokens": 700, "completion_tokens": 120}}
        ).encode()
        router = MixedRouter(
            FakePool([w]),
            FakePolicy(),
            nats_client=FakeNats([(TYPE_DATA, None, body), (TYPE_DONE, 200, b"")]),
        )

        isl_before = histogram_totals(metrics.input_sequence_tokens, router="mixed")
        osl_before = histogram_totals(metrics.output_sequence_tokens, router="mixed")
        response = await router.dispatch({"model": "m"}, stream=False)
        assert response.status_code == 200

        isl_after = histogram_totals(metrics.input_sequence_tokens, router="mixed")
        osl_after = histogram_totals(metrics.output_sequence_tokens, router="mixed")
        assert isl_after[0] - isl_before[0] == pytest.approx(700.0)
        assert osl_after[0] - osl_before[0] == pytest.approx(120.0)
        await router.aclose()

    async def test_no_ttft_is_recorded_for_a_non_streaming_reply(self):
        # The whole response lands at once, so there is no first-token boundary
        # to time; reporting the full duration as TTFT would poison the average.
        w = worker()
        body = json.dumps({"usage": {"prompt_tokens": 700, "completion_tokens": 120}}).encode()
        router = MixedRouter(
            FakePool([w]),
            FakePolicy(),
            nats_client=FakeNats([(TYPE_DATA, None, body), (TYPE_DONE, 200, b"")]),
        )
        before = histogram_totals(metrics.time_to_first_token_seconds, router="mixed")
        await router.dispatch({"model": "m"}, stream=False)
        after = histogram_totals(metrics.time_to_first_token_seconds, router="mixed")
        assert after == before
        await router.aclose()
