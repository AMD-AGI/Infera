###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""fp8 KV passthrough — CPU-only correctness gates for the kvd L3
connector.

The CORRECT design (per the user) is byte PASSTHROUGH: when vLLM is
launched with ``--kv-cache-dtype fp8`` the paged KV tensors are already
fp8 (uint8 / float8_e4m3*). The connector must move those exact bytes to
L3 and back UNCHANGED — NO connector-side quantize on save, NO dequant on
load, NO per-chunk scale. vLLM owns k_scale/v_scale at attention time.

These tests assert:
  (a) an fp8 (uint8) KV tensor survives pack → unpack → scatter
      BIT-IDENTICAL (the passthrough guarantee);
  (b) bf16 still round-trips bit-identical (back-compat);
  (c) the abandoned quantize/dequant/scale machinery is GONE — no
      ``scale`` header field, no ``INFERA_KVD_L3_FP8`` knob, no
      ``fp8_l3`` module, no ``_scatter_dtype_for`` helper.

All pure-CPU: the Python scatter/gather fallback in
``triton_kv_gather`` is the path exercised when ``device.type ==
"cpu"`` (and is the byte-exact oracle for the Triton kernel). No GPU,
no kvd daemon, no docker.
"""

from __future__ import annotations

import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
torch_skip = pytest.mark.skipif(torch_spec is None, reason="torch not installed")


# ----------------------------------------------------------------------
# Helpers — build a paged KV layout + pack/scatter it the same way the
# connector does, but standalone (no daemon, no async path).
# ----------------------------------------------------------------------


def _fp8_or_uint8_dtype():
    """Prefer a real fp8 dtype when the torch build exposes one;
    otherwise fall back to uint8 (which is how vLLM stores fp8 KV under
    the hood, and what _torch_dtype_to_str maps to "fp8_e4m3"). Either
    way it is a 1-byte element, which is the property under test."""
    import torch

    for name in ("float8_e4m3fnuz", "float8_e4m3fn"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return torch.uint8


def _make_regular_layers(num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, salt):
    """Regular attention paged layout: [2, num_blocks, block_size,
    num_kv_heads, head_dim]. Filled with a deterministic byte pattern so
    bit-equality is meaningful even for 1-byte dtypes."""
    import torch

    layers = {}
    for li in range(num_layers):
        # Build the raw byte pattern in uint8 first, then bit-cast to the
        # target dtype via .view — this guarantees every fp8 byte is a
        # well-defined value (avoids NaN/inf surprises from random fp8).
        # Size the byte buffer by element_count * itemsize so the .view to a
        # multi-byte dtype (bf16=2B) yields exactly element_count elements.
        elems = 2 * num_blocks * block_size * num_kv_heads * head_dim
        n = elems * torch.empty(0, dtype=dtype).element_size()
        flat = torch.empty(n, dtype=torch.uint8)
        for i in range(n):
            flat[i] = (li * 31 + i * 7 + salt) & 0xFF
        t = flat.view(dtype).reshape(2, num_blocks, block_size, num_kv_heads, head_dim)
        layers[f"layer.{li}"] = t
    return layers


def _pack_and_scatter_roundtrip(dtype):
    """Gather a deterministic paged KV tensor into KV_2LTD staging, pack
    it to a v2 chunk blob, unpack, and scatter into a fresh paged buffer
    — all on CPU via the Python oracle path. Returns (src_layers,
    dst_layers, header, producer_pages, consumer_pages) for the caller
    to byte-compare."""
    import numpy as _np
    import torch

    from infera.engine.vllm.kvd_connector import _torch_dtype_to_str
    from infera.engine.vllm.packed_format import (
        ChunkHeader,
        pack_chunk_header,
        unpack_chunk,
    )
    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 16, 16
    num_kv_heads, head_dim = 2, 8
    hidden_dim = num_kv_heads * head_dim
    chunk_tokens = block_size * 2  # 2 pages / chunk
    num_kv_channels = 2

    src_layers = _make_regular_layers(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, dtype, salt=11
    )
    src_list = [src_layers[f"layer.{i}"] for i in range(num_layers)]

    # Producer pages (deliberately non-contiguous) → staging via gather.
    producer_pages = ((3,), (5,))
    layer_to_group = [0] * num_layers
    staging = torch.zeros((num_kv_channels, num_layers, chunk_tokens, hidden_dim), dtype=dtype)
    kv_chunk_gather(staging, src_list, producer_pages, layer_to_group, block_size, use_triton=False)

    # Pack the staging payload exactly like the connector save path:
    # raw bytes are the dtype's bytes, header dtype name from the live
    # dtype (fp8 → "fp8_e4m3"), dtype_bytes drives payload size.
    dtype_str = _torch_dtype_to_str(dtype)
    header = ChunkHeader(
        version=2,
        chunk_tokens=chunk_tokens,
        block_size=block_size,
        num_layers=num_layers,
        layer_names=tuple(f"layer.{i}" for i in range(num_layers)),
        hidden_dim=hidden_dim,
        dtype=dtype_str,
        cache_group_id=0,
        num_kv_channels=num_kv_channels,
    )
    # The payload is the raw element bytes — view as uint8 (1 byte each
    # for fp8, no scaling, no conversion).
    payload = staging.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    assert len(payload) == header.payload_bytes, (
        f"payload {len(payload)} != header.payload_bytes {header.payload_bytes} "
        f"(dtype_bytes={header.dtype_bytes}) — fp8 must be 1 byte/element"
    )
    blob = pack_chunk_header(header) + payload

    # Unpack and re-materialize the staging tensor the SAME way the load
    # path does: frombuffer(uint8) → .view(cache_dtype). cache dtype ==
    # stored dtype (passthrough), so no dequant.
    got_header, got_payload = unpack_chunk(blob)
    arr = _np.frombuffer(got_payload, dtype=_np.uint8)
    decoded = (
        torch.from_numpy(arr.copy())
        .view(dtype)
        .reshape(num_kv_channels, num_layers, chunk_tokens, hidden_dim)
    )

    # Scatter into a fresh, DIFFERENT set of pages.
    dst_layers = {
        f"layer.{i}": torch.zeros((2, num_blocks, block_size, num_kv_heads, head_dim), dtype=dtype)
        for i in range(num_layers)
    }
    dst_list = [dst_layers[f"layer.{i}"] for i in range(num_layers)]
    consumer_pages = ((7,), (11,))
    kv_chunk_scatter(
        decoded, dst_list, consumer_pages, layer_to_group, block_size, use_triton=False
    )
    return src_layers, dst_layers, got_header, producer_pages, consumer_pages


def _assert_bit_identical(src_layers, dst_layers, producer_pages, consumer_pages):
    """Every producer page must equal the corresponding consumer page,
    compared on the RAW bytes (works for fp8 where == on NaN would lie)."""
    import torch

    for li, lname in enumerate(src_layers):
        for (src_pg,), (tgt_pg,) in zip(producer_pages, consumer_pages):
            src_bytes = src_layers[lname][:, src_pg].contiguous().view(torch.uint8)
            tgt_bytes = dst_layers[lname][:, tgt_pg].contiguous().view(torch.uint8)
            assert torch.equal(src_bytes, tgt_bytes), (
                f"layer {li}: producer page {src_pg} bytes != consumer page "
                f"{tgt_pg} bytes — passthrough was NOT byte-identical"
            )


# ----------------------------------------------------------------------
# (a) fp8 round-trips bit-identical (passthrough, no scale)
# ----------------------------------------------------------------------


@torch_skip
def test_fp8_chunk_roundtrip_bit_identical():
    """An fp8 (uint8 / float8_e4m3*) KV tensor survives
    gather → pack → unpack → scatter with bit-identical bytes. No
    quantize, no dequant, no scale: the exact bytes vLLM wrote come
    back unchanged."""
    dtype = _fp8_or_uint8_dtype()
    src, dst, header, pp, cp = _pack_and_scatter_roundtrip(dtype)
    # Header must advertise the fp8 1-byte payload.
    assert header.dtype == "fp8_e4m3"
    assert header.dtype_bytes == 1
    _assert_bit_identical(src, dst, pp, cp)


# ----------------------------------------------------------------------
# (b) bf16 still round-trips (back-compat)
# ----------------------------------------------------------------------


@torch_skip
def test_bf16_chunk_roundtrip_bit_identical():
    """bf16 chunks load exactly as before — the fp8 work must not
    regress the dominant 2-byte path."""
    import torch

    src, dst, header, pp, cp = _pack_and_scatter_roundtrip(torch.bfloat16)
    assert header.dtype == "bf16"
    assert header.dtype_bytes == 2
    _assert_bit_identical(src, dst, pp, cp)


# ----------------------------------------------------------------------
# (c) the abandoned quantize/dequant/scale machinery is GONE
# ----------------------------------------------------------------------


def test_no_scale_field_on_chunk_header():
    """The per-chunk `scale` header field (quantize approach) is
    reverted: ChunkHeader has no `scale`, and a header round-trips
    without one."""
    from infera.engine.vllm.packed_format import (
        ChunkHeader,
        pack_chunk_header,
        unpack_chunk,
    )

    fields = set(ChunkHeader.__dataclass_fields__)
    assert "scale" not in fields, "ChunkHeader.scale must be reverted"

    header = ChunkHeader(
        version=2,
        chunk_tokens=64,
        block_size=16,
        num_layers=4,
        layer_names=tuple(f"layer.{i}" for i in range(4)),
        hidden_dim=128,
        dtype="fp8_e4m3",
        cache_group_id=0,
        num_kv_channels=2,
    )
    payload = b"\x00" * header.payload_bytes
    got_header, _ = unpack_chunk(pack_chunk_header(header) + payload)
    assert got_header == header
    assert not hasattr(got_header, "scale")


def test_no_quantize_machinery_in_source():
    """No knob, no fp8_l3 module, no quantize/dequant helpers remain in
    the connector — guards against the abandoned approach creeping back."""
    import pathlib

    import infera.engine.vllm.kvd_connector as conn

    # The fp8_l3 module must be deleted.
    assert importlib.util.find_spec("infera.engine.vllm.fp8_l3") is None, (
        "infera.engine.vllm.fp8_l3 must be deleted (quantize approach abandoned)"
    )

    src = pathlib.Path(conn.__file__).read_text()
    for needle in (
        "INFERA_KVD_L3_FP8",
        "_l3_fp8",
        "fp8_l3",
        "quantize_chunk_to_fp8",
        "dequantize_fp8_chunk",
        "_scatter_dtype_for",
    ):
        assert needle not in src, (
            f"abandoned quantize machinery still present in kvd_connector.py: {needle!r}"
        )

    # The connector also must not expose the reverted attribute.
    assert not hasattr(conn.InferaKvdConnector, "_scatter_dtype_for"), (
        "_scatter_dtype_for helper must be removed"
    )
