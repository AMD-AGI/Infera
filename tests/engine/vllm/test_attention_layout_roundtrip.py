###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Per-attention-layout KV write/load correctness gates for the kvd L3 connector.

End-to-end output comparison can NOT verify cache write/load correctness: on
reload vLLM recomputes the dropped sub-chunk tail, and MoE recompute is
nondeterministic, so reload output diverges from a fresh compute even when the
reloaded KV is byte-exact (observed for bf16 AND fp8 alike — see PR #122). The
deterministic gate is a synthetic gather(save) -> pack -> unpack -> scatter(load)
round-trip, one case per KV cache layout the connector must handle:

  * regular attention   [2, num_blocks, block_size, H, D]  (K/V outermost)
  * FullAttention 0.23  [num_blocks, 2, block_size, H, D]  (K/V interleaved)
  * MLA latent          [num_blocks, block_size, hidden]   (1-channel)
  * MLA + fp8 packed     same shape, uint8/fp8 bytes        (fp8_ds_mla latent)

Plus: a DSA mixed group (main MLA latent + sparse-attention indexer,
glm_moe_dsa / deepseek_v32) must be WHOLE-GROUP-SKIPPED — never offloaded, since
L3 reuse of a DSA group corrupts sparse attention regardless of indexer handling.

All pure-CPU via the Python oracle path (use_triton=False). No GPU, no daemon.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

torch_skip = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)


def _fp8_or_uint8():
    import torch

    for name in ("float8_e4m3fn", "float8_e4m3fnuz"):
        dt = getattr(torch, name, None)
        if dt is not None:
            return dt
    return torch.uint8


def _fill(dtype, elems, salt):
    """Deterministic byte pattern -> view as dtype (well-defined for fp8)."""
    import torch

    n = elems * torch.empty(0, dtype=dtype).element_size()
    flat = torch.empty(n, dtype=torch.uint8)
    for i in range(n):
        flat[i] = (i * 7 + salt * 131) & 0xFF
    return flat.view(dtype)


# ---- per-layout paged-tensor builders + page extractors -------------------
# Each builder returns {layer_name: tensor}; each extractor returns the raw
# uint8 bytes of one page (block) for byte-comparison.


def _make_regular(nl, nb, bs, h, d, dtype):
    return {
        f"l.{i}": _fill(dtype, 2 * nb * bs * h * d, salt=i).reshape(2, nb, bs, h, d)
        for i in range(nl)
    }


def _page_regular(t, bid):
    return t[:, bid].contiguous().view(-1).view(_u8())  # [2, bs, h, d]


def _make_interleaved(nl, nb, bs, h, d, dtype):
    return {
        f"l.{i}": _fill(dtype, nb * 2 * bs * h * d, salt=i).reshape(nb, 2, bs, h, d)
        for i in range(nl)
    }


def _page_interleaved(t, bid):
    return t[bid].contiguous().view(-1).view(_u8())  # [2, bs, h, d]


def _make_mla(nl, nb, bs, hidden, dtype):
    return {
        f"l.{i}": _fill(dtype, nb * bs * hidden, salt=i).reshape(nb, bs, hidden) for i in range(nl)
    }


def _page_mla(t, bid):
    return t[bid].contiguous().view(-1).view(_u8())  # [bs, hidden]


def _u8():
    import torch

    return torch.uint8


def _roundtrip(make_fn, page_fn, *, nl, nb, bs, hidden, num_kv_channels, dtype, empty_shape):
    """gather -> pack -> unpack -> scatter, all CPU/Python. Returns
    (src_layers, dst_layers, producer_pages, consumer_pages)."""
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

    chunk_tokens = bs * 2  # 2 pages / chunk
    src_layers = make_fn(dtype)
    src_list = [src_layers[f"l.{i}"] for i in range(nl)]
    layer_to_group = [0] * nl
    producer_pages = ((3,), (5,))

    staging = torch.zeros((num_kv_channels, nl, chunk_tokens, hidden), dtype=dtype)
    kv_chunk_gather(staging, src_list, producer_pages, layer_to_group, bs, use_triton=False)

    header = ChunkHeader(
        version=2,
        chunk_tokens=chunk_tokens,
        block_size=bs,
        num_layers=nl,
        layer_names=tuple(f"l.{i}" for i in range(nl)),
        hidden_dim=hidden,
        dtype=_torch_dtype_to_str(dtype),
        cache_group_id=0,
        num_kv_channels=num_kv_channels,
    )
    payload = staging.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    assert len(payload) == header.payload_bytes, (
        f"payload {len(payload)} != header.payload_bytes {header.payload_bytes}"
    )
    blob = pack_chunk_header(header) + payload

    got_header, got_payload = unpack_chunk(blob)
    arr = np.frombuffer(got_payload, dtype=np.uint8)
    decoded = (
        torch.from_numpy(arr.copy()).view(dtype).reshape(num_kv_channels, nl, chunk_tokens, hidden)
    )

    dst_layers = {f"l.{i}": torch.zeros(empty_shape, dtype=dtype) for i in range(nl)}
    dst_list = [dst_layers[f"l.{i}"] for i in range(nl)]
    consumer_pages = ((7,), (11,))
    kv_chunk_scatter(decoded, dst_list, consumer_pages, layer_to_group, bs, use_triton=False)
    return src_layers, dst_layers, producer_pages, consumer_pages


def _assert_pages_equal(src_layers, dst_layers, page_fn, producer_pages, consumer_pages, nl):
    for i in range(nl):
        s = src_layers[f"l.{i}"]
        d = dst_layers[f"l.{i}"]
        for (pbid,), (cbid,) in zip(producer_pages, consumer_pages):
            sb = bytes(page_fn(s, pbid).numpy().tobytes())
            db = bytes(page_fn(d, cbid).numpy().tobytes())
            assert sb == db, f"layer {i}: producer page {pbid} != consumer page {cbid}"


# ------------------------------- tests -------------------------------------
NL, NB, BS, H, D = 4, 16, 16, 2, 8
HID = H * D  # 16


@torch_skip
@pytest.mark.parametrize("dtype_name", ["bf16", "fp8"])
def test_roundtrip_regular(dtype_name):
    import torch

    dt = torch.bfloat16 if dtype_name == "bf16" else _fp8_or_uint8()
    src, dst, pp, cp = _roundtrip(
        lambda dtype: _make_regular(NL, NB, BS, H, D, dtype),
        _page_regular,
        nl=NL,
        nb=NB,
        bs=BS,
        hidden=HID,
        num_kv_channels=2,
        dtype=dt,
        empty_shape=(2, NB, BS, H, D),
    )
    _assert_pages_equal(src, dst, _page_regular, pp, cp, NL)


@torch_skip
@pytest.mark.parametrize("dtype_name", ["bf16", "fp8"])
def test_roundtrip_fullattention_interleaved(dtype_name):
    """vLLM 0.23 FullAttention KV layout [num_blocks, 2, block_size, H, D]."""
    import torch

    dt = torch.bfloat16 if dtype_name == "bf16" else _fp8_or_uint8()
    src, dst, pp, cp = _roundtrip(
        lambda dtype: _make_interleaved(NL, NB, BS, H, D, dtype),
        _page_interleaved,
        nl=NL,
        nb=NB,
        bs=BS,
        hidden=HID,
        num_kv_channels=2,
        dtype=dt,
        empty_shape=(NB, 2, BS, H, D),
    )
    _assert_pages_equal(src, dst, _page_interleaved, pp, cp, NL)


@torch_skip
@pytest.mark.parametrize("layout", ["regular", "interleaved"])
def test_roundtrip_gqa_fp8_minimax_dims(layout):
    """GQA fp8 KV at MiniMax-M2.5 head dims (8 kv-heads x 128 = 1024 hidden),
    both the [2, nb, bs, H, D] and the [nb, 2, bs, H, D] layouts vLLM emits.

    This is the exact packed layout `register_kv_caches` now OFFLOADS for a
    plain GQA fp8 cache (hidden == (num_key_value_heads // tp) * head_dim, no
    interleaved scale) — see `_expected_plain_gqa_hidden`. A plain fp8 cast is
    a contiguous per-tensor-scale byte run, so the gather/pack/unpack/scatter
    round-trips it byte-exact regardless of head dims. Pinning MiniMax's real
    dims guards the production shape (observed live: (64465, 2, 16, 8, 128)).
    """

    mh, md = 8, 128  # MiniMax-M2.5: num_key_value_heads x head_dim -> hidden 1024
    dt = _fp8_or_uint8()
    if layout == "regular":
        make_fn, page_fn, empty = _make_regular, _page_regular, (2, NB, BS, mh, md)
    else:
        make_fn, page_fn, empty = _make_interleaved, _page_interleaved, (NB, 2, BS, mh, md)
    src, dst, pp, cp = _roundtrip(
        lambda dtype: make_fn(NL, NB, BS, mh, md, dtype),
        page_fn,
        nl=NL,
        nb=NB,
        bs=BS,
        hidden=mh * md,
        num_kv_channels=2,
        dtype=dt,
        empty_shape=empty,
    )
    _assert_pages_equal(src, dst, page_fn, pp, cp, NL)


@torch_skip
@pytest.mark.parametrize("dtype_name", ["bf16", "fp8"])
def test_roundtrip_mla(dtype_name):
    """MLA combined latent [num_blocks, block_size, hidden]; fp8 == fp8_ds_mla."""
    import torch

    dt = torch.bfloat16 if dtype_name == "bf16" else _fp8_or_uint8()
    src, dst, pp, cp = _roundtrip(
        lambda dtype: _make_mla(NL, NB, BS, HID, dtype),
        _page_mla,
        nl=NL,
        nb=NB,
        bs=BS,
        hidden=HID,
        num_kv_channels=1,
        dtype=dt,
        empty_shape=(NB, BS, HID),
    )
    _assert_pages_equal(src, dst, _page_mla, pp, cp, NL)


@torch_skip
def test_dsa_mixed_group_detected():
    """A group mixing the DSA sparse-attention indexer with the main MLA latent
    must be recognised so register_kv_caches whole-group-skips it."""
    from infera.engine.vllm.kvd_connector import _is_dsa_indexer_group

    main_only = ["model.layers.0.self_attn.attn", "model.layers.1.self_attn.attn"]
    indexer = "model.layers.0.self_attn.indexer.k_cache"
    assert _is_dsa_indexer_group([*main_only, indexer]) is True
    assert _is_dsa_indexer_group(main_only) is False  # non-DSA: strict no-op
