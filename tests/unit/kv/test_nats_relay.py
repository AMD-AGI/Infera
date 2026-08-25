###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""The relay has to read what its own engine writes.

vLLM and ATOM emit tagged-map KV events; SGLang emits tagged arrays. The relay
decodes batches for one purpose only -- maintaining the KV bucket a cold router
bootstraps from -- and the live forward onto NATS happens *before* the decode,
on the raw bytes. So a decoder aimed at the wrong family produces no visible
symptom at all: events flow, the router updates, and only the bucket stays empty
forever, which is indistinguishable from a worker that has served no traffic.
That is how a hardcoded vLLM decoder survived three releases on an
SGLang-default deployment. These tests are the alarm that was missing.
"""

from __future__ import annotations

import asyncio
import logging

import pytest
from msgspec.msgpack import Decoder, Encoder

from infera.common.worker_pool import EngineType
from infera.kv import nats_relay as relay_mod
from infera.kv.nats_relay import KvEventNatsRelay
from infera.router.kv_event.events import (
    AllBlocksCleared,
    KVEventBatch,
    SglangAllBlocksCleared,
    SglangBlockStored,
    SglangKVEventBatch,
)

_ENC = Encoder()


def _sglang_stored(*, parent: int | None = None) -> bytes:
    return _ENC.encode(
        SglangKVEventBatch(
            ts=1.0,
            events=[
                SglangBlockStored(
                    block_hashes=[111],
                    parent_block_hash=parent,
                    token_ids=[1, 2, 3, 4],
                    block_size=4,
                    lora_id=None,
                )
            ],
            attn_dp_rank=0,
        )
    )


def _relay(engine) -> KvEventNatsRelay:
    return KvEventNatsRelay(
        worker_id="w1",
        engine_zmq_endpoint="tcp://0.0.0.0:5557",
        engine=engine,
        block_size=4,
    )


class _FakeSocket:
    """Hands out each payload once, then lets ``_loop`` fall out of its while."""

    def __init__(self, relay: KvEventNatsRelay, payloads: list[bytes]) -> None:
        self._relay = relay
        self._payloads = list(payloads)

    async def recv_multipart(self):
        payload = self._payloads.pop(0)
        if not self._payloads:
            self._relay._closing = True
        return [b"kv-events", payload]


class _FakeBus:
    def __init__(self) -> None:
        self.published: list[bytes] = []

    async def js_publish(self, subject, payload):
        self.published.append(payload)


async def _drive(relay: KvEventNatsRelay, payloads: list[bytes]) -> _FakeBus:
    bus = _FakeBus()
    relay._bus = bus
    await relay._loop(0, _FakeSocket(relay, payloads))
    return bus


@pytest.mark.asyncio
async def test_an_sglang_worker_gets_a_decoder_that_reads_sglang(caplog):
    relay = _relay(EngineType.SGLANG)
    with caplog.at_level(logging.WARNING, logger="infera.kv.nats_relay"):
        await _drive(relay, [_sglang_stored()])
    assert relay._decode_failures == 0, caplog.text
    # The point of decoding at all: the per-rank view the bucket is written from.
    assert relay._sub.view_for(0), "a decoded BlockStored must land in the view"


@pytest.mark.asyncio
async def test_the_wrong_family_decoder_is_reported_not_swallowed(caplog):
    """The regression itself: a vLLM decoder against an SGLang stream.

    Note what still works -- the raw batch reaches NATS regardless. That is
    precisely why this needs a log line to be noticeable at all.
    """
    relay = _relay(EngineType.SGLANG)
    relay._decoder = Decoder(type=KVEventBatch)
    with caplog.at_level(logging.WARNING, logger="infera.kv.nats_relay"):
        bus = await _drive(relay, [_sglang_stored()])
    assert relay._decode_failures == 1
    assert not relay._sub.view_for(0)
    assert bus.published, "the live forward is unaffected -- hence the need to say so"
    assert "cannot decode" in caplog.text


def test_the_decoder_follows_the_engine():
    assert _relay(EngineType.SGLANG)._decoder.type is SglangKVEventBatch
    assert _relay(EngineType.VLLM)._decoder.type is KVEventBatch
    assert _relay(EngineType.ATOM)._decoder.type is KVEventBatch
    # A worker that did not say defaults to SGLang, the default deployment.
    assert _relay(None)._decoder.type is SglangKVEventBatch


def test_decode_failures_back_off_instead_of_logging_per_event(caplog):
    relay = _relay(EngineType.SGLANG)
    with caplog.at_level(logging.WARNING, logger="infera.kv.nats_relay"):
        for _ in range(30):
            relay._note_decode_failure(0, ValueError("nope"))
    # First, then 10th, then 100th (never reached) -- a broken decoder fails at
    # event rate, and logging each one buries the message it is trying to send.
    assert len(caplog.records) == 2
    assert relay._decode_failures == 30


# --- the signal the startup flush waits on ------------------------------------


@pytest.mark.asyncio
async def test_a_clear_raises_the_flag_the_flush_loop_waits_on():
    """``anchor_kv_chain`` retries until this fires.

    Without it the worker could only flush and hope: ZMQ ``connect()`` is
    asynchronous, so a flush issued before this subscription attached is lost
    exactly the way the original anchor was, and looks identical to success.
    """
    relay = _relay(EngineType.SGLANG)
    assert not relay.cleared_observed.is_set()
    await _drive(
        relay, [_ENC.encode(SglangKVEventBatch(ts=1.0, events=[SglangAllBlocksCleared()]))]
    )
    assert relay.cleared_observed.is_set()


@pytest.mark.asyncio
async def test_ordinary_traffic_does_not_raise_it():
    """It has to mean *this* flush landed, not merely that events are arriving
    -- otherwise the retry loop would stop on the first batch either way."""
    relay = _relay(EngineType.SGLANG)
    await _drive(relay, [_sglang_stored()])
    assert not relay.cleared_observed.is_set()


@pytest.mark.asyncio
async def test_a_vllm_clear_counts_too():
    """The two families are different classes with the same meaning; matching
    only SGLang's would hang the vLLM worker's flush loop through every retry."""
    relay = _relay(EngineType.VLLM)
    await _drive(relay, [_ENC.encode(KVEventBatch(ts=1.0, events=[AllBlocksCleared()]))])
    assert relay.cleared_observed.is_set()


@pytest.mark.asyncio
async def test_the_flag_survives_the_events_that_follow_it():
    """The flush's observer may not be scheduled before more traffic arrives."""
    relay = _relay(EngineType.SGLANG)
    await _drive(
        relay,
        [
            _ENC.encode(SglangKVEventBatch(ts=1.0, events=[SglangAllBlocksCleared()])),
            _sglang_stored(),
        ],
    )
    assert relay.cleared_observed.is_set()
    assert isinstance(relay.cleared_observed, asyncio.Event)


class _FakeKv:
    def __init__(self) -> None:
        self.puts: list[bytes] = []

    async def put(self, key, value):
        self.puts.append(value)


@pytest.mark.asyncio
async def test_the_tail_of_a_burst_reaches_the_bucket(monkeypatch):
    """The coalescing interval used to swallow the newest state indefinitely.

    ``_maybe_write_bucket`` was only ever reached from ``_loop``, so an event
    landing inside the interval marked the rank dirty and returned, and the
    write it deferred happened on the *next* event -- which, at the tail of a
    burst, is whenever traffic resumes. Minutes of the freshest view went
    unmirrored while the bucket held a snapshot from before the burst, so a
    cold-starting router seeded stale, and the router's own coverage window
    lapsed against a relay that was working fine.
    """
    monkeypatch.setattr(relay_mod, "_BUCKET_WRITE_INTERVAL_S", 0.05)
    relay = _relay(EngineType.SGLANG)
    relay._kv = kv = _FakeKv()

    # Two events back to back: the first writes, the second is inside the
    # interval and only sets the dirty bit.
    await _drive(relay, [_sglang_stored(), _sglang_stored(parent=111)])
    assert len(kv.puts) == 1
    assert relay._dirty[0] is True, "the newest view is the part left unwritten"

    # Nothing more arrives on this rank. The drain is what comes back for it.
    relay._closing = False
    task = asyncio.create_task(relay._drain_dirty())
    for _ in range(100):
        await asyncio.sleep(0.01)
        if len(kv.puts) > 1:
            break
    relay._closing = True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(kv.puts) == 2, "the tail never reached the bucket"
    assert relay._dirty[0] is False


@pytest.mark.asyncio
async def test_the_drain_writes_nothing_when_no_rank_is_dirty(monkeypatch):
    """It runs for the life of the worker, so a quiet rank must cost a dict
    lookup and not a bucket round-trip per tick."""
    monkeypatch.setattr(relay_mod, "_BUCKET_WRITE_INTERVAL_S", 0.02)
    relay = _relay(EngineType.SGLANG)
    relay._kv = kv = _FakeKv()

    task = asyncio.create_task(relay._drain_dirty())
    await asyncio.sleep(0.15)
    relay._closing = True
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert kv.puts == []
