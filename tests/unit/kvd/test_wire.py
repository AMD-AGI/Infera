###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/kvd/wire.py — IPC frame encoding and decoding."""

from __future__ import annotations

import asyncio

import pytest

from infera.kvd.wire import (
    LENGTH_PREFIX_BYTES,
    Clear,
    ClearAck,
    ErrorMessage,
    Exists,
    ExistsResponse,
    Get,
    GetResponse,
    Hello,
    HelloAck,
    OpCode,
    Set,
    SetAck,
    Stats,
    StatsResponse,
    decode,
    encode,
    read_frame,
    validate_retention,
    write_frame,
)

# ----------------------------------------------------------------------
# Round-trip: every message type
# ----------------------------------------------------------------------


def test_hello_round_trip():
    msg = Hello(client_id="alice", protocol_version=1)
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_hello_with_prefers_shared_arena():
    """Hello carries an opt-in flag for shared-arena negotiation.
    Default False keeps existing clients silent."""
    msg = Hello(client_id="alice", prefers_shared_arena=True)
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg
    assert decoded.prefers_shared_arena is True


def test_hello_ack_round_trip():
    msg = HelloAck(server_id="kvd-abc", protocol_version=1)
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_hello_ack_with_shared_arena():
    """HelloAck carries (arena_size, slot_size, server_pid) when the
    server is shared-arena capable AND the client opted in."""
    msg = HelloAck(
        server_id="kvd-abc",
        shared_arena=(32 * 1024**3, 4 * 1024**2, 12345),
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg
    # msgpack decodes tuples as lists; we re-tupleize so equality
    # and type-tests downstream behave consistently.
    assert isinstance(decoded.shared_arena, tuple)


def test_hello_ack_default_shared_arena_none():
    msg = HelloAck(server_id="kvd-abc")
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded.shared_arena is None


def test_get_round_trip():
    msg = Get(key=b"\x01\x02\x03\x04", model="m1", compat_key="ck1")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_get_response_with_value():
    msg = GetResponse(value=b"\xff" * 64, retention="long")
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_get_response_miss_carries_none():
    msg = GetResponse(value=None, retention=None)
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded.value is None
    assert decoded.retention is None


def test_set_round_trip_with_metadata():
    msg = Set(
        key=b"hash-key",
        value=b"x" * 16,
        retention="long",
        model="m",
        compat_key="ck",
        metadata={"layer": 7, "dtype": "bf16"},
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_set_ack_with_rejection_reason():
    msg = SetAck(accepted=False, reason="would_displace_higher_priority")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_batch_get_round_trip():
    """BatchGet carries a list of 8-byte keys + the namespace. Wire
    round-trip preserves order (the response must match position-wise)."""
    from infera.kvd.wire import BatchGet

    msg = BatchGet(keys=[b"k1" + b"\x00" * 6, b"k2" + b"\x00" * 6], model="m", compat_key="ck")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_batch_get_response_with_mixed_hits():
    """Response interleaves bytes-or-None per the request order.
    Position-aligned semantics — `values[i] is None` means key i missed."""
    from infera.kvd.wire import BatchGetResponse

    msg = BatchGetResponse(
        values=[b"\xff" * 8, None, b"\xaa" * 16],
        retentions=["short", None, "long"],
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg
    assert decoded.values[1] is None


def test_batch_get_empty_keys_round_trip():
    """Empty BatchGet is legal at the wire (the client short-circuits
    before sending, but the daemon must tolerate it)."""
    from infera.kvd.wire import BatchGet

    msg = BatchGet(keys=[], model="", compat_key="")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_batch_get_response_empty_is_legal():
    from infera.kvd.wire import BatchGetResponse

    msg = BatchGetResponse(values=[], retentions=[])
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_batch_set_round_trip():
    """BatchSet carries parallel arrays of (key, value, retention,
    metadata) plus the namespace. Wire round-trip preserves order +
    arity — server expects position-aligned arrays."""
    from infera.kvd.wire import BatchSet

    msg = BatchSet(
        keys=[b"k1" + b"\x00" * 6, b"k2" + b"\x00" * 6],
        values=[b"v1" * 4, b"v2" * 4],
        retentions=["short", "long"],
        metadatas=[{"layer_count": 32}, {}],
        model="m",
        compat_key="ck",
    )
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_batch_set_ack_round_trip_mixed_results():
    """Daemon may accept some and reject others — parallel arrays."""
    from infera.kvd.wire import BatchSetAck

    msg = BatchSetAck(
        accepted=[True, False, True],
        reasons=[None, "would_displace_higher_priority", None],
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg
    assert decoded.accepted[1] is False


def test_batch_set_empty_round_trip():
    """Empty BatchSet is legal at the wire (client short-circuits
    before sending, but daemon must tolerate it)."""
    from infera.kvd.wire import BatchSet, BatchSetAck

    req = BatchSet(keys=[], values=[], retentions=[], metadatas=[], model="", compat_key="")
    assert decode(encode(req)[LENGTH_PREFIX_BYTES:]) == req
    ack = BatchSetAck(accepted=[], reasons=[])
    assert decode(encode(ack)[LENGTH_PREFIX_BYTES:]) == ack


# ----------------------------------------------------------------------
# TTL + ephemeral retention round-trip
# ----------------------------------------------------------------------


def test_set_ttl_seconds_round_trip():
    """Set.ttl_seconds must round-trip cleanly (None default + finite)."""
    from infera.kvd.wire import RETENTION_SHORT, Set

    msg = Set(key=b"k" * 8, value=b"v" * 32, retention=RETENTION_SHORT, ttl_seconds=3600.0)
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg
    # None (no TTL) is the default and must serialize cleanly.
    msg_no_ttl = Set(key=b"k" * 8, value=b"v" * 32, retention=RETENTION_SHORT)
    decoded = decode(encode(msg_no_ttl)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg_no_ttl
    assert decoded.ttl_seconds is None


def test_batch_set_ttls_seconds_round_trip():
    """BatchSet.ttls_seconds: optional parallel array. None on the
    request = no-TTL-on-any-item. When present, each entry's TTL
    individually controllable."""
    from infera.kvd.wire import RETENTION_EPHEMERAL, RETENTION_SHORT, BatchSet

    msg = BatchSet(
        keys=[b"a" * 8, b"b" * 8, b"c" * 8],
        values=[b"v1" * 8, b"v2" * 8, b"v3" * 8],
        retentions=[RETENTION_SHORT, RETENTION_EPHEMERAL, RETENTION_SHORT],
        metadatas=[{}, {}, {}],
        ttls_seconds=[60.0, None, 3600.0],
    )
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg
    # ttls_seconds omitted entirely must default to None.
    no_ttl_batch = BatchSet(
        keys=[b"x" * 8], values=[b"v"], retentions=[RETENTION_SHORT], metadatas=[{}]
    )
    decoded = decode(encode(no_ttl_batch)[LENGTH_PREFIX_BYTES:])
    assert decoded == no_ttl_batch
    assert decoded.ttls_seconds is None


def test_ephemeral_retention_constant_exposed():
    """The ephemeral retention class must be a valid retention string.
    If a future refactor renames/removes it, callers that built JSON
    requests with the literal `"ephemeral"` would silently get
    validation errors at the daemon. Pin the name."""
    from infera.kvd.wire import RETENTION_EPHEMERAL, validate_retention

    assert RETENTION_EPHEMERAL == "ephemeral"
    validate_retention("ephemeral")  # must not raise


# ----------------------------------------------------------------------
# PrefetchHint
# ----------------------------------------------------------------------


def test_prefetch_hint_round_trip():
    """PrefetchHint carries N block hashes + namespace + deadline.
    Fire-and-forget — there's no response opcode. The wire test
    pins that the request shape round-trips cleanly."""
    from infera.kvd.wire import PrefetchHint

    msg = PrefetchHint(
        keys=[b"a" * 8, b"b" * 8, b"c" * 8],
        model="MiniMax-M2.5",
        compat_key="fp/abcd",
        deadline_ms=500,
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg
    assert decoded.deadline_ms == 500


def test_prefetch_hint_empty_keys_round_trip():
    """Empty key list is legal on the wire (clients short-circuit
    before sending, but the daemon must tolerate it for forward-
    compat with router code that filters keys before dispatch)."""
    from infera.kvd.wire import PrefetchHint

    msg = PrefetchHint(keys=[], model="", compat_key="", deadline_ms=1000)
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_prefetch_hint_default_deadline():
    """When the caller doesn't pass `deadline_ms` the wire default
    is 1000ms. Pin this so a future refactor doesn't silently
    change the default (which would alter TTL on warmed entries)."""
    from infera.kvd.wire import PrefetchHint

    msg = PrefetchHint(keys=[b"k" * 8])
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded.deadline_ms == 1000


def test_exists_round_trip_empty_keys_ok():
    msg = Exists(keys=[], model="m", compat_key="ck")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_exists_response_round_trip():
    msg = ExistsResponse(present=[True, False, True])
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_clear_default_clears_all():
    msg = Clear()
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_clear_ack_round_trip():
    msg = ClearAck(cleared_entries=42)
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_stats_request_round_trip():
    msg = Stats()
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert isinstance(decoded, Stats)


def test_stats_response_round_trip():
    msg = StatsResponse(
        entries=10,
        host_bytes=4096,
        spillover_bytes=0,
        long_bytes=0,
        gets_total=100,
        sets_total=50,
        hits_total=90,
        misses_total=10,
        evictions_total=5,
    )
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_error_message_round_trip():
    msg = ErrorMessage(code="bad_op", message="unknown opcode foo")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


# ----------------------------------------------------------------------
# Frame structure
# ----------------------------------------------------------------------


def test_encoded_frame_starts_with_length_prefix():
    msg = Hello(client_id="x")
    frame = encode(msg)
    assert len(frame) > LENGTH_PREFIX_BYTES
    declared_len = int.from_bytes(frame[:LENGTH_PREFIX_BYTES], "big")
    assert declared_len == len(frame) - LENGTH_PREFIX_BYTES


def test_encoded_bytes_field_stays_bytes():
    """msgpack distinguishes str and bytes; we want bytes to stay bytes
    so the engine doesn't have to b64-encode raw KV bytes."""
    msg = Set(key=b"\x00\xff", value=b"hello", retention="short")
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert isinstance(decoded.key, bytes)
    assert isinstance(decoded.value, bytes)
    assert decoded.key == b"\x00\xff"
    assert decoded.value == b"hello"


# ----------------------------------------------------------------------
# Error paths
# ----------------------------------------------------------------------


def test_decode_unknown_op_raises():
    import msgpack

    body = msgpack.packb({"op": "frobnicate"}, use_bin_type=True)
    with pytest.raises(ValueError, match="unknown op"):
        decode(body)


def test_decode_missing_op_raises():
    import msgpack

    body = msgpack.packb({"hello": "world"}, use_bin_type=True)
    with pytest.raises(ValueError, match="not an op dict"):
        decode(body)


def test_decode_bad_payload_raises():
    import msgpack

    body = msgpack.packb({"op": "get", "wrong_field": "x"}, use_bin_type=True)
    with pytest.raises(ValueError, match="bad payload"):
        decode(body)


def test_validate_retention_accepts_known():
    assert validate_retention("none") == "none"
    assert validate_retention("short") == "short"
    assert validate_retention("long") == "long"


def test_validate_retention_rejects_unknown():
    with pytest.raises(ValueError, match="unknown retention"):
        validate_retention("forever")


def test_opcode_enum_values_are_stable():
    """The opcode strings are wire-protocol — changing them is a
    breaking change. Lock them in as a test."""
    assert OpCode.HELLO.value == "hello"
    assert OpCode.GET.value == "get"
    assert OpCode.SET.value == "set"
    assert OpCode.EXISTS.value == "exists"
    assert OpCode.CLEAR.value == "clear"
    assert OpCode.STATS.value == "stats"
    assert OpCode.HELLO_ACK.value == "hello_ack"
    assert OpCode.GET_RESPONSE.value == "get_response"
    assert OpCode.SET_ACK.value == "set_ack"
    assert OpCode.EXISTS_RESPONSE.value == "exists_response"
    assert OpCode.CLEAR_ACK.value == "clear_ack"
    assert OpCode.STATS_RESPONSE.value == "stats_response"
    assert OpCode.ERROR.value == "error"


# ----------------------------------------------------------------------
# read_frame / write_frame async helpers
# ----------------------------------------------------------------------


class _FakeStream:
    """In-memory bidirectional stream for testing read/write frame
    helpers without spinning up a real socket."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._closed = False

    # Writer methods used by write_frame
    def write(self, data: bytes) -> None:
        self._buf.extend(data)

    async def drain(self) -> None:
        pass

    # Reader methods used by read_frame
    async def readexactly(self, n: int) -> bytes:
        if len(self._buf) < n:
            raise asyncio.IncompleteReadError(partial=bytes(self._buf), expected=n)
        out = bytes(self._buf[:n])
        del self._buf[:n]
        return out


@pytest.mark.asyncio
async def test_write_and_read_frame_round_trip():
    stream = _FakeStream()
    sent = Set(key=b"\x01", value=b"hello", retention="long")
    await write_frame(stream, sent)
    received = await read_frame(stream)
    assert received == sent


@pytest.mark.asyncio
async def test_read_frame_propagates_incomplete_read():
    stream = _FakeStream()
    # Only partial header → IncompleteReadError
    with pytest.raises(asyncio.IncompleteReadError):
        await read_frame(stream)


@pytest.mark.asyncio
async def test_read_frame_zero_length_body_raises():
    stream = _FakeStream()
    stream._buf.extend((0).to_bytes(LENGTH_PREFIX_BYTES, "big"))
    with pytest.raises(ValueError, match="empty frame body"):
        await read_frame(stream)


# ----------------------------------------------------------------------
# Shared-arena response types
# ----------------------------------------------------------------------


def test_get_shared_response_round_trip():
    """GetSharedResponse carries (offset, length, version) — no value
    bytes. The client (which mmapped the arena at handshake) reads
    bytes from its own mmap using the seqlock protocol."""
    from infera.kvd.wire import GetSharedResponse

    msg = GetSharedResponse(
        slot_offset=12345,
        length=4096,
        version=4,
        retention="short",
        slot_size=4128,
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_batch_get_shared_response_round_trip():
    """BatchGetSharedResponse uses parallel arrays so partial-hit
    batches don't waste bytes encoding miss-padding. `offsets[i] < 0`
    or `lengths[i] == 0` means miss for position i."""
    from infera.kvd.wire import BatchGetSharedResponse

    msg = BatchGetSharedResponse(
        offsets=[100, -1, 200],
        lengths=[64, 0, 128],
        versions=[2, 0, 4],
        retentions=["short", None, "long"],
        slot_size=256,
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_get_shared_response_no_retention():
    """Retention is optional — `None` round-trips correctly."""
    from infera.kvd.wire import GetSharedResponse

    msg = GetSharedResponse(slot_offset=0, length=10, version=2)
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


# ----------------------------------------------------------------------
# Save-side CopyFree: lease + commit two-phase set
# ----------------------------------------------------------------------


def test_set_reserve_round_trip():
    """SetReserve carries only the intended payload size — the key is
    revealed at commit so the engine can reuse a lease for any key."""
    from infera.kvd.wire import SetReserve

    msg = SetReserve(size=4096)
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_set_reserve_response_accept_round_trip():
    """Accept: nonzero lease_token, real slot_id, payload_offset past
    the slot header, payload_max_size = slot_size - header."""
    from infera.kvd.wire import SetReserveResponse

    msg = SetReserveResponse(
        lease_token=42,
        slot_id=7,
        payload_offset=7 * 4128 + 16,
        payload_max_size=4128 - 16,
        reason="",
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_set_reserve_response_reject_round_trip():
    """Reject: lease_token=0, slot_id=-1, reason carries stable token."""
    from infera.kvd.wire import SetReserveResponse

    msg = SetReserveResponse(
        lease_token=0,
        slot_id=-1,
        payload_offset=0,
        payload_max_size=0,
        reason="arena_full_no_evictable",
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_set_commit_round_trip_with_all_fields():
    """SetCommit names the key, length, model, compat_key, retention,
    ttl — every field round-trips."""
    from infera.kvd.wire import SetCommit

    msg = SetCommit(
        lease_token=42,
        key=b"\xde\xad\xbe\xef" * 2,
        length=4096,
        model="qwen3-0.6b",
        compat_key="rank0:tp1",
        retention="long",
        ttl_seconds=3600,
    )
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg


def test_set_commit_defaults_round_trip():
    """Most fields default — `model=""`, `compat_key=""`,
    `retention="default"`, `ttl_seconds=0`."""
    from infera.kvd.wire import SetCommit

    msg = SetCommit(lease_token=1, key=b"k", length=64)
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert decoded == msg
    assert decoded.retention == "default"
    assert decoded.ttl_seconds == 0


def test_set_commit_response_round_trip():
    from infera.kvd.wire import SetCommitResponse

    msg = SetCommitResponse(accepted=True, reason="")
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg

    msg2 = SetCommitResponse(accepted=False, reason="unknown_lease")
    assert decode(encode(msg2)[LENGTH_PREFIX_BYTES:]) == msg2


def test_set_cancel_round_trip():
    from infera.kvd.wire import SetCancel

    msg = SetCancel(lease_token=99)
    assert decode(encode(msg)[LENGTH_PREFIX_BYTES:]) == msg


def test_set_cancel_response_round_trip():
    """No-field dataclass — encodes as `{"op": "set_cancel_response"}`."""
    from infera.kvd.wire import SetCancelResponse

    msg = SetCancelResponse()
    decoded = decode(encode(msg)[LENGTH_PREFIX_BYTES:])
    assert isinstance(decoded, SetCancelResponse)
    assert decoded == msg
