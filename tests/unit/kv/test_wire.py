###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
from __future__ import annotations

import pytest

from infera.kv.wire import (
    EVENT_VERSION,
    EventBatch,
    EventType,
    decode_batch,
    decode_topic,
    encode_batch,
    encode_topic,
    make_cleared_event,
    make_removed_event,
    make_stored_event,
)


def test_event_type_enum_values() -> None:
    assert EventType.STORED.value == "stored"
    assert EventType.REMOVED.value == "removed"
    assert EventType.CLEARED.value == "cleared"


def test_topic_round_trip() -> None:
    name = "Qwen/Qwen3.6-27B"
    assert decode_topic(encode_topic(name)) == name


def test_make_stored_event_fields() -> None:
    e = make_stored_event(
        sequence_hash=123,
        block_hash=456,
        parent_sequence_hash=None,
        tier="device",
        role="indexable",
        group_idx=0,
    )
    assert e.type == EventType.STORED
    assert e.sequence_hash == 123
    assert e.block_hash == 456
    assert e.parent_sequence_hash is None
    assert e.tier == "device"


def test_make_removed_event_fields() -> None:
    e = make_removed_event(sequence_hash=999, tier="host")
    assert e.type == EventType.REMOVED
    assert e.sequence_hash == 999
    assert e.tier == "host"
    assert e.block_hash is None
    assert e.parent_sequence_hash is None


def test_make_cleared_event_fields() -> None:
    e = make_cleared_event(scope="all")
    assert e.type == EventType.CLEARED
    assert e.scope == "all"


def test_normalized_event_msgpack_dict_omits_none() -> None:
    e = make_removed_event(sequence_hash=1, tier="device")
    d = e.to_msgpack_dict()
    assert "block_hash" not in d
    assert "parent_sequence_hash" not in d
    assert "scope" not in d
    assert d["sequence_hash"] == 1
    assert d["tier"] == "device"


def test_batch_round_trip_stored() -> None:
    batch = EventBatch(
        publisher_id="w1",
        publisher_type="worker",
        model_name="Qwen3.6",
        compat_key="abcdef0123456789",
        index_block_size=64,
        batch_id=42,
        events=(
            make_stored_event(
                sequence_hash=111, block_hash=222, parent_sequence_hash=None, tier="device"
            ),
            make_stored_event(
                sequence_hash=333, block_hash=444, parent_sequence_hash=111, tier="device"
            ),
        ),
    )
    payload = encode_batch(batch)
    decoded = decode_batch(payload)
    assert decoded.publisher_id == "w1"
    assert decoded.publisher_type == "worker"
    assert decoded.model_name == "Qwen3.6"
    assert decoded.compat_key == "abcdef0123456789"
    assert decoded.index_block_size == 64
    assert decoded.batch_id == 42
    assert decoded.event_version == EVENT_VERSION
    assert len(decoded.events) == 2
    assert decoded.events[0].type == EventType.STORED
    assert decoded.events[0].sequence_hash == 111
    assert decoded.events[1].parent_sequence_hash == 111


def test_batch_round_trip_mixed_events() -> None:
    batch = EventBatch(
        publisher_id="infera-kvd-node-1",
        publisher_type="pool",
        model_name="Qwen3.6",
        compat_key="abcdef0123456789",
        index_block_size=64,
        batch_id=0,
        events=(
            make_stored_event(
                sequence_hash=1,
                block_hash=10,
                parent_sequence_hash=None,
                tier="host",
                pool_id="infera-kvd-node-1",
                pool_type="infera-kvd",
            ),
            make_removed_event(sequence_hash=1, tier="host", pool_id="infera-kvd-node-1"),
            make_cleared_event(scope="compat_key:abcdef0123456789"),
        ),
    )
    decoded = decode_batch(encode_batch(batch))
    assert len(decoded.events) == 3
    assert decoded.events[0].pool_id == "infera-kvd-node-1"
    assert decoded.events[0].pool_type == "infera-kvd"
    assert decoded.events[1].type == EventType.REMOVED
    assert decoded.events[2].type == EventType.CLEARED
    assert decoded.events[2].scope == "compat_key:abcdef0123456789"


def test_batch_round_trip_empty_events() -> None:
    batch = EventBatch(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=64,
        batch_id=0,
        events=(),
    )
    decoded = decode_batch(encode_batch(batch))
    assert decoded.events == ()


def test_decode_rejects_unknown_event_version() -> None:
    import msgpack

    future_payload = msgpack.packb(
        {
            "v": EVENT_VERSION + 99,
            "publisher_id": "w1",
            "publisher_type": "worker",
            "model_name": "m",
            "compat_key": "ck",
            "index_block_size": 64,
            "batch_id": 0,
            "events": [],
        },
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="newer than supported"):
        decode_batch(future_payload)


def test_decode_rejects_missing_version() -> None:
    import msgpack

    payload = msgpack.packb({"publisher_id": "w1"}, use_bin_type=True)
    with pytest.raises(ValueError, match="event_version"):
        decode_batch(payload)


def test_decode_rejects_non_dict_root() -> None:
    import msgpack

    payload = msgpack.packb([1, 2, 3], use_bin_type=True)
    with pytest.raises(ValueError, match="expected dict"):
        decode_batch(payload)


def test_decode_rejects_unknown_event_type() -> None:
    import msgpack

    payload = msgpack.packb(
        {
            "v": EVENT_VERSION,
            "publisher_id": "w1",
            "publisher_type": "worker",
            "model_name": "m",
            "compat_key": "ck",
            "index_block_size": 64,
            "batch_id": 0,
            "events": [{"type": "bogus"}],
        },
        use_bin_type=True,
    )
    with pytest.raises(ValueError, match="unknown event type"):
        decode_batch(payload)


def test_mm_extra_round_trip() -> None:
    """Multimodal extra metadata survives encode/decode."""
    e = make_stored_event(
        sequence_hash=1,
        block_hash=2,
        parent_sequence_hash=None,
        tier="device",
        mm_extra=[{"mm_hash": "0xdead", "slots": [3, 4, 5]}],
    )
    batch = EventBatch(
        publisher_id="w1",
        publisher_type="worker",
        model_name="m",
        compat_key="ck",
        index_block_size=64,
        batch_id=0,
        events=(e,),
    )
    decoded = decode_batch(encode_batch(batch))
    assert decoded.events[0].mm_extra == [{"mm_hash": "0xdead", "slots": [3, 4, 5]}]
