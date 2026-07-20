###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Pack/unpack KV cache chunks for the vLLM connector.

vLLM-8 chunked-fusion wire format (v2). One kvd_key carries N
consecutive vLLM pages × all layers within one kv_cache_group,
serialized as ``[2, num_layers, chunk_tokens, hidden_dim]`` (the
``KV_2LTD`` layout — same shape LMCache uses for its CPU/GDS pool).

Design rationale: per the MI355X perf bench (issue
``AMD-AGI/Infera#44``), per-page kvd
keys produced 0.6–4.3 MiB files spread across 30–78 per-layer copy
operations, all converging to ~1–4 ms per load regardless of
storage medium (GDS, mmap, CPU-bounce). The bottleneck is per-call
overhead, not bandwidth. Coalescing into one ``packed_chunks_to_save``
entry per N-page chunk replaces that with a single Triton kernel +
single POSIX write, cutting per-call overhead by num_layers × N.

## Wire layout

A chunk blob is::

    [u32 LE: header_length]
    [msgpack body]
    [KV_2LTD payload bytes]

The msgpack body is::

    {
      "v": 2,
      "chunk_tokens": int,         # = block_size × N
      "block_size": int,           # vLLM page size in tokens
      "num_layers": int,
      "layer_names": [str, ...],   # in blob order (= layer_idx → name)
      "hidden_dim": int,           # per-token KV element count
      "dtype": str,                # "bf16" | "fp16" | "fp8_e4m3" | "fp32"
      "cache_group_id": int        # HMA cache_group this chunk belongs to
    }

The payload is ``2 × num_layers × chunk_tokens × hidden_dim × dtype_bytes``
bytes, laid out as a contiguous ``KV_2LTD`` tensor that the worker's
Triton scatter kernel re-distributes into per-layer paged KV slots.

## Format guarantees

- **Order-preserving**: ``layer_names`` lists the per-layer order in
  the payload tensor; the worker iterates them to dispatch.
- **Self-describing**: ``hidden_dim`` + ``dtype`` + ``chunk_tokens``
  fully determine the payload size; ``ChunkHeader.payload_bytes``
  derives it.
- **HMA-aware**: ``cache_group_id`` identifies which kv_cache_group
  the chunk's layers belong to; HMA models emit one chunk per
  (chunk-window, cache_group), so the same prompt content produces
  multiple files (one per group) with distinct kvd_keys.
- **Versioned**: ``v=2`` is the only supported version. Older blobs
  raise ``PackedFormatError`` on unpack → caller treats as cache
  miss. v1 (per-page packed) was deleted in vLLM-9.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import msgpack

_CHUNK_FORMAT_VERSION = 2
_HEADER_LEN_PREFIX = 4  # bytes, u32 LE


class PackedFormatError(Exception):
    """Raised when a chunk blob can't be parsed (truncation, version
    mismatch, malformed header). Callers MUST treat this as a cache
    miss — never as a corrupt-read into the paged buffer."""


@dataclass(frozen=True)
class ChunkHeader:
    """Decoded v2 chunk header. The payload bytes live immediately
    after the header (at file offset ``4 + header_len``) and decode as
    a flat ``[2, num_layers, chunk_tokens, hidden_dim]`` tensor in
    ``dtype``. ``layer_names`` is the per-layer name in blob order so
    consumers can sanity-check against their registered KV caches.
    """

    version: int  # always _CHUNK_FORMAT_VERSION (= 2)
    chunk_tokens: int  # 1 page = block_size tokens; N pages = chunk_tokens
    block_size: int  # vLLM page size in tokens
    num_layers: int
    layer_names: tuple[str, ...]
    hidden_dim: int  # per-token KV element count (= num_kv_heads_per_rank × head_dim, or kv_lora_rank + qk_rope_head_dim for MLA)
    dtype: str  # one of {"bf16", "fp16", "fp8_e4m3", "fp32"}; element bytes derived
    cache_group_id: int  # which kv_cache_group these layers belong to (HMA-aware)
    num_kv_channels: int = 2  # 2 = K+V (regular attention), 1 = combined latent (MLA)
    # Absolute byte offset within the chunk file where the payload
    # starts. 0 means "right after the msgpack body" (legacy v2 behavior
    # — payload starts at HEADER_LEN_PREFIX + msgpack_len). Non-zero
    # means the writer padded between header and payload (typically to
    # round payload_byte_offset up to a 4 KiB boundary so hipFile/
    # cuFile reads land on aligned offsets). New readers honor this
    # field; old readers see legacy chunks unchanged (field defaults to
    # 0 when the msgpack body doesn't include it).
    payload_byte_offset: int = 0

    @property
    def dtype_bytes(self) -> int:
        return {
            "bf16": 2,
            "fp16": 2,
            "fp8_e4m3": 1,
            "fp32": 4,
        }.get(self.dtype, 2)

    @property
    def payload_bytes(self) -> int:
        """Total bytes for the KV_2LTD (or KV_1LTD for MLA) payload."""
        return (
            self.num_kv_channels
            * self.num_layers
            * self.chunk_tokens
            * self.hidden_dim
            * self.dtype_bytes
        )


def pack_chunk_header(header: ChunkHeader) -> bytes:
    """Serialize a v2 header (length prefix + msgpack body). Caller
    appends the KV_2LTD payload bytes (which it owns; pack_chunk_header
    doesn't allocate the chunk-sized buffer)."""
    body = msgpack.packb(
        {
            "v": _CHUNK_FORMAT_VERSION,
            "chunk_tokens": int(header.chunk_tokens),
            "block_size": int(header.block_size),
            "num_layers": int(header.num_layers),
            "layer_names": list(header.layer_names),
            "hidden_dim": int(header.hidden_dim),
            "dtype": str(header.dtype),
            "cache_group_id": int(header.cache_group_id),
            "num_kv_channels": int(header.num_kv_channels),
            "payload_byte_offset": int(header.payload_byte_offset),
        },
        use_bin_type=True,
    )
    return struct.pack("<I", len(body)) + body


def pack_chunk_header_aligned(
    header: ChunkHeader,
    align: int = 4096,
) -> tuple[bytes, ChunkHeader]:
    """Like :func:`pack_chunk_header` but pads the result with NUL
    bytes so the *payload* (which the caller will append after the
    returned bytes) starts on an ``align``-byte boundary.

    Returns ``(padded_header_bytes, updated_header)`` — the caller
    appends the payload bytes immediately after ``padded_header_bytes``
    and the file looks like::

        [4-byte u32 msgpack_len] [msgpack body] [NUL pad] [payload]
        |<- HEADER_LEN_PREFIX ->|<-- msgpack_len -->|
        |<------------ payload_byte_offset --------->|

    ``updated_header.payload_byte_offset`` is set to the absolute byte
    offset where the payload starts (= len(padded_header_bytes)). Pass
    the returned ``updated_header`` to recipients that want to know
    the on-disk layout; the bytes themselves embed it too via the
    msgpack ``payload_byte_offset`` field.

    Two-pass dance: the field's value depends on the msgpack body's
    length, which depends on the field's serialized length, which is
    fixed-width for sufficiently-large ints in msgpack but the safest
    way is pack-twice. With a 4 KiB alignment and ~600 byte headers
    the overhead is negligible (~3.5 KiB of NULs per ~34 MiB chunk).
    """
    # First pass: tentative pack to discover header size.
    body_v0 = msgpack.packb(
        {
            "v": _CHUNK_FORMAT_VERSION,
            "chunk_tokens": int(header.chunk_tokens),
            "block_size": int(header.block_size),
            "num_layers": int(header.num_layers),
            "layer_names": list(header.layer_names),
            "hidden_dim": int(header.hidden_dim),
            "dtype": str(header.dtype),
            "cache_group_id": int(header.cache_group_id),
            "num_kv_channels": int(header.num_kv_channels),
            "payload_byte_offset": 0,  # placeholder, finalised below
        },
        use_bin_type=True,
    )
    unpadded_len = _HEADER_LEN_PREFIX + len(body_v0)
    aligned_offset = (unpadded_len + align - 1) & ~(align - 1)

    # Second pass: pack with the real offset value.
    final_header = ChunkHeader(
        version=header.version,
        chunk_tokens=header.chunk_tokens,
        block_size=header.block_size,
        num_layers=header.num_layers,
        layer_names=header.layer_names,
        hidden_dim=header.hidden_dim,
        dtype=header.dtype,
        cache_group_id=header.cache_group_id,
        num_kv_channels=header.num_kv_channels,
        payload_byte_offset=aligned_offset,
    )
    body_v1 = msgpack.packb(
        {
            "v": _CHUNK_FORMAT_VERSION,
            "chunk_tokens": int(final_header.chunk_tokens),
            "block_size": int(final_header.block_size),
            "num_layers": int(final_header.num_layers),
            "layer_names": list(final_header.layer_names),
            "hidden_dim": int(final_header.hidden_dim),
            "dtype": str(final_header.dtype),
            "cache_group_id": int(final_header.cache_group_id),
            "num_kv_channels": int(final_header.num_kv_channels),
            "payload_byte_offset": int(final_header.payload_byte_offset),
        },
        use_bin_type=True,
    )
    new_unpadded_len = _HEADER_LEN_PREFIX + len(body_v1)
    if new_unpadded_len > aligned_offset:
        # Field grew across packs (rare with msgpack's varint encoding;
        # only happens at byte-count thresholds). Recompute aligned
        # offset and re-pack one more time — guaranteed to fit since
        # the field width is now stable.
        aligned_offset = (new_unpadded_len + align - 1) & ~(align - 1)
        final_header = ChunkHeader(
            version=header.version,
            chunk_tokens=header.chunk_tokens,
            block_size=header.block_size,
            num_layers=header.num_layers,
            layer_names=header.layer_names,
            hidden_dim=header.hidden_dim,
            dtype=header.dtype,
            cache_group_id=header.cache_group_id,
            num_kv_channels=header.num_kv_channels,
            payload_byte_offset=aligned_offset,
        )
        body_v1 = msgpack.packb(
            {
                "v": _CHUNK_FORMAT_VERSION,
                "chunk_tokens": int(final_header.chunk_tokens),
                "block_size": int(final_header.block_size),
                "num_layers": int(final_header.num_layers),
                "layer_names": list(final_header.layer_names),
                "hidden_dim": int(final_header.hidden_dim),
                "dtype": str(final_header.dtype),
                "cache_group_id": int(final_header.cache_group_id),
                "num_kv_channels": int(final_header.num_kv_channels),
                "payload_byte_offset": int(final_header.payload_byte_offset),
            },
            use_bin_type=True,
        )
    out = struct.pack("<I", len(body_v1)) + body_v1
    pad = aligned_offset - len(out)
    if pad > 0:
        out = out + (b"\x00" * pad)
    return out, final_header


def unpack_chunk(
    blob: bytes | memoryview,
    *,
    header_only: bool = False,
) -> tuple[ChunkHeader, bytes | memoryview]:
    """Parse a v2 chunk blob. Returns ``(header, payload_view)`` where
    the payload is a zero-copy slice of ``blob`` (memoryview when
    input is memoryview; bytes when input is bytes). Caller decodes
    the payload bytes into a torch tensor of shape
    ``[2, num_layers, chunk_tokens, hidden_dim]`` and runs the
    scatter gather.

    ``header_only=True`` parses just the header and SKIPS payload-size
    validation, returning ``(header, b"")``. The GPU-direct loader uses
    this: it preads only the first ~64 KiB to learn the geometry, then
    DMAs the (130+ MiB) payload straight to device — it never holds the
    full blob in host memory, so payload validation would always (and
    wrongly) fail.

    Raises ``PackedFormatError`` on:
      - blob too short
      - header not msgpack-decodable
      - version != 2 (v1 blobs hit this branch — caller treats as
        cache miss per the migration policy)
      - payload bytes don't match header's declared size (unless
        ``header_only``)
    """
    if len(blob) < _HEADER_LEN_PREFIX:
        raise PackedFormatError(
            f"chunk blob too short for header length prefix ({len(blob)} < {_HEADER_LEN_PREFIX})"
        )
    (header_len,) = struct.unpack("<I", blob[:_HEADER_LEN_PREFIX])
    header_start = _HEADER_LEN_PREFIX
    header_end = header_start + header_len
    if header_end > len(blob):
        raise PackedFormatError(
            f"chunk blob too short for declared header: have "
            f"{len(blob) - header_start} bytes, need {header_len}"
        )
    try:
        body = msgpack.unpackb(blob[header_start:header_end], raw=False)
    except (ValueError, msgpack.UnpackException) as exc:
        raise PackedFormatError(f"chunk header not valid msgpack: {exc}") from exc
    if not isinstance(body, dict):
        raise PackedFormatError(f"chunk header msgpack is not a dict: {type(body).__name__}")
    version = body.get("v")
    if version != _CHUNK_FORMAT_VERSION:
        # v1 blobs (or any unknown version) hit this branch and the
        # caller treats the lookup as a miss. We deliberately don't
        # back-decode v1 per the clean-break migration policy.
        raise PackedFormatError(
            f"unsupported chunk format version {version!r} "
            f"(this build only reads v{_CHUNK_FORMAT_VERSION})"
        )
    try:
        header = ChunkHeader(
            version=version,
            chunk_tokens=int(body["chunk_tokens"]),
            block_size=int(body["block_size"]),
            num_layers=int(body["num_layers"]),
            layer_names=tuple(str(n) for n in body["layer_names"]),
            hidden_dim=int(body["hidden_dim"]),
            dtype=str(body["dtype"]),
            cache_group_id=int(body.get("cache_group_id", 0)),
            # num_kv_channels defaults to 2 for backward compat with
            # blobs written before MLA support — those were always
            # 2-channel (K+V split) regular attention.
            num_kv_channels=int(body.get("num_kv_channels", 2)),
            # payload_byte_offset defaults to 0 = "right after msgpack
            # body" (legacy v2 unpadded). Non-zero means the writer
            # padded for 4 KiB alignment (cuFile / hipFile-async).
            payload_byte_offset=int(body.get("payload_byte_offset", 0)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PackedFormatError(f"chunk header missing/invalid field: {exc}") from exc
    if len(header.layer_names) != header.num_layers:
        raise PackedFormatError(
            f"chunk header: num_layers={header.num_layers} but "
            f"layer_names has {len(header.layer_names)} entries"
        )
    if header.num_kv_channels not in (1, 2):
        raise PackedFormatError(
            f"chunk header: num_kv_channels={header.num_kv_channels} "
            f"must be 1 (MLA) or 2 (regular attention)"
        )
    # Payload starts at header.payload_byte_offset (when non-zero,
    # accounts for header-padding NULs) or at header_end (legacy v2).
    payload_start = header.payload_byte_offset if header.payload_byte_offset > 0 else header_end
    if header_only:
        # Geometry-only parse for the GPU-direct loader — payload lives
        # past the pread window and goes straight to device, so don't
        # validate or slice it.
        return header, b""
    if payload_start > len(blob):
        raise PackedFormatError(
            f"chunk payload_byte_offset={payload_start} exceeds blob size {len(blob)}"
        )
    payload_view = blob[payload_start:]
    expected = header.payload_bytes
    if len(payload_view) != expected:
        raise PackedFormatError(
            f"chunk payload size mismatch: header declared {expected} "
            f"bytes ([{header.num_kv_channels},{header.num_layers},"
            f"{header.chunk_tokens},{header.hidden_dim}]×"
            f"{header.dtype_bytes}), blob has {len(payload_view)} after "
            f"header (payload_start={payload_start})"
        )
    return header, payload_view


def chunk_header_size_overhead(num_layers: int) -> int:
    """Approximate header-only bytes for a v2 chunk. Payload bytes
    are model-dependent (see ``ChunkHeader.payload_bytes``). Used for
    sizing prediction in benches."""
    if num_layers <= 0:
        return _HEADER_LEN_PREFIX + 64
    # 4-byte length prefix + msgpack envelope (~80 bytes constant
    # for v2's larger field set) + ~40 bytes per layer name.
    return _HEADER_LEN_PREFIX + 80 + 40 * num_layers
