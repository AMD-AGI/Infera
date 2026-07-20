###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/engine/vllm/packed_format.py — v2 chunked-fusion
wire format (pack_chunk_header / unpack_chunk / ChunkHeader)."""

from __future__ import annotations

import struct

import pytest

from infera.engine.vllm.packed_format import (
    ChunkHeader,
    PackedFormatError,
    chunk_header_size_overhead,
    pack_chunk_header,
    unpack_chunk,
)

# ----------------------------------------------------------------------
# v2 chunk format (LMCache-style fusion, vLLM-8)
# ----------------------------------------------------------------------


def _chunk_header(
    *, chunk_tokens=64, block_size=16, num_layers=4, hidden_dim=128, dtype="bf16", cache_group_id=0
):
    return ChunkHeader(
        version=2,
        chunk_tokens=chunk_tokens,
        block_size=block_size,
        num_layers=num_layers,
        layer_names=tuple(f"layer.{i}" for i in range(num_layers)),
        hidden_dim=hidden_dim,
        dtype=dtype,
        cache_group_id=cache_group_id,
    )


def test_chunk_round_trip_minimal():
    """Header serializes, payload appends as-is, unpack yields a
    matching header and payload view."""
    header = _chunk_header()
    payload = bytes(range(256)) * (header.payload_bytes // 256)
    assert len(payload) == header.payload_bytes
    blob = pack_chunk_header(header) + payload
    got_header, got_payload = unpack_chunk(blob)
    assert got_header == header
    assert bytes(got_payload) == payload


def test_chunk_round_trip_via_memoryview():
    """When the input is a memoryview (mmap'd file), the payload
    slice returned is a zero-copy memoryview — confirm bytes match."""
    header = _chunk_header(num_layers=8, chunk_tokens=128, hidden_dim=64)
    payload = b"\xab" * header.payload_bytes
    blob = pack_chunk_header(header) + payload
    got_header, got_payload = unpack_chunk(memoryview(blob))
    assert isinstance(got_payload, memoryview)
    assert bytes(got_payload) == payload
    assert got_header.num_layers == 8


def test_chunk_rejects_v1_blob_as_unsupported():
    """v1 blobs (or any version != 2) hit unpack_chunk and raise —
    caller treats as cache miss per the migration policy. v1 was
    deleted in vLLM-9 so we manually construct a v1-shaped blob to
    verify the version check still rejects it cleanly."""
    import msgpack

    body = msgpack.packb({"v": 1, "layers": [["layer.0", 16]]}, use_bin_type=True)
    v1_blob = struct.pack("<I", len(body)) + body + b"x" * 16
    with pytest.raises(PackedFormatError, match="version"):
        unpack_chunk(v1_blob)


def test_chunk_rejects_truncated_payload():
    header = _chunk_header()
    payload = b"\x00" * (header.payload_bytes - 8)  # missing last 8 bytes
    blob = pack_chunk_header(header) + payload
    with pytest.raises(PackedFormatError, match="payload size mismatch"):
        unpack_chunk(blob)


def test_chunk_rejects_extra_trailing_bytes():
    header = _chunk_header()
    payload = b"\x00" * header.payload_bytes + b"EXTRA"
    blob = pack_chunk_header(header) + payload
    with pytest.raises(PackedFormatError, match="payload size mismatch"):
        unpack_chunk(blob)


def test_chunk_rejects_inconsistent_num_layers_and_names():
    """num_layers and len(layer_names) must agree — defensive check."""
    import msgpack

    body = msgpack.packb(
        {
            "v": 2,
            "chunk_tokens": 64,
            "block_size": 16,
            "num_layers": 4,  # says 4
            "layer_names": ["a", "b"],  # only 2 names
            "hidden_dim": 32,
            "dtype": "bf16",
            "cache_group_id": 0,
        },
        use_bin_type=True,
    )
    blob = struct.pack("<I", len(body)) + body + b"\x00" * (2 * 4 * 64 * 32 * 2)
    with pytest.raises(PackedFormatError, match="layer_names has"):
        unpack_chunk(blob)


def test_chunk_payload_size_formula():
    """payload_bytes = 2 × num_layers × chunk_tokens × hidden_dim × dtype_bytes."""
    for dtype, db in (("bf16", 2), ("fp16", 2), ("fp8_e4m3", 1), ("fp32", 4)):
        h = _chunk_header(num_layers=3, chunk_tokens=64, hidden_dim=128, dtype=dtype)
        assert h.payload_bytes == 2 * 3 * 64 * 128 * db, dtype


def test_chunk_header_size_overhead_handles_zero_layers():
    """Edge: an empty-layer chunk should still report SOME baseline
    header overhead (caller uses this for sizing)."""
    assert chunk_header_size_overhead(0) > 0


def test_chunk_header_size_overhead_grows_with_layers():
    """More layers → more header bytes (each layer_name string costs)."""
    assert chunk_header_size_overhead(80) > chunk_header_size_overhead(8)


def test_chunk_header_dtype_default_falls_back_to_2_bytes():
    """Unknown dtype string defaults to 2 bytes/element (bf16-equivalent).
    Defensive — protects against operator typos."""
    h = ChunkHeader(
        version=2,
        chunk_tokens=64,
        block_size=16,
        num_layers=2,
        layer_names=("a", "b"),
        hidden_dim=32,
        dtype="i-made-this-up",
        cache_group_id=0,
    )
    assert h.dtype_bytes == 2


def test_chunk_header_serialization_is_deterministic():
    """Same header in → same bytes out. Required for content-addressed
    file naming via SHA on the blob."""
    header = _chunk_header()
    a = pack_chunk_header(header)
    b = pack_chunk_header(header)
    assert a == b


def test_chunk_header_serialization_changes_on_field_change():
    """Sanity that the msgpack body actually reflects the fields."""
    h1 = _chunk_header(chunk_tokens=64)
    h2 = _chunk_header(chunk_tokens=128)
    assert pack_chunk_header(h1) != pack_chunk_header(h2)


def test_chunk_header_is_frozen():
    """ChunkHeader is immutable so consumers can stash it."""
    import dataclasses

    h = _chunk_header()
    with pytest.raises(dataclasses.FrozenInstanceError):
        h.chunk_tokens = 999  # type: ignore[misc]
