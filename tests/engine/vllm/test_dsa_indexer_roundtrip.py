###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""DSA sparse-attention **indexer key cache** round-trip correctness gate.

Motivation (see the kvd-L3 × attention-architecture notes, §5): on a DSA model
(deepseek_v32 / GLM-5-DSA / DeepSeek-V4) an external-L3 prefix hit skips prefill,
so the reused prefix's `indexer.k_cache` must be RESTORED, not recomputed. The
claim under test: the indexer key IS cacheable, because its per-token layout is
self-contained.

Indexer K-cache layout (vLLM 0.23, `deepseek_v4/attention.py`:
`head_dim bytes = 128 fp8 + 4 fp32 scale = 132`): each token is a CONTIGUOUS
132-byte run — 128 fp8 index bytes immediately followed by a 4-byte fp32 (ue8m0)
scale. Because the fp8 data and its scale are co-packed per token, a raw-byte
gather/scatter round-trips BOTH as a coherent unit — no scale desync, unlike the
main K-cache whose scales live in a separate per-block region.

This is the deterministic, CPU-only gate the notes' §5.5 calls for at the unit
level (no GPU, no model forward, not confounded by the reload-tail recompute
that makes E2E output diffs unusable): build a synthetic indexer cache with a
DISTINCT (fp8-region, scale-region) pattern per token, run the connector's
gather -> pack -> unpack -> scatter, and assert every token's 132 bytes — fp8
region [0:128] AND scale region [128:132] — survive byte-exact into a different
physical block.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

torch_skip = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)

INDEXER_HIDDEN = 132  # 128 fp8 index + 4-byte fp32 scale, co-packed/token
FP8_BYTES = 128
SCALE_OFF = 128  # scale region = bytes [128:132]


def _make_indexer_layers(num_layers, num_blocks, block_size):
    """Synthetic indexer.k_cache per layer: uint8 [num_blocks, block_size, 132].
    Each token's 132 bytes are unique AND its scale region [128:132] is a
    distinct, non-trivial fp32 value, so a lost/duplicated scale is caught."""
    import torch

    layers = {}
    for li in range(num_layers):
        t = torch.empty((num_blocks, block_size, INDEXER_HIDDEN), dtype=torch.uint8)
        for b in range(num_blocks):
            for p in range(block_size):
                tok = b * block_size + p
                # fp8 index region: deterministic distinct bytes
                for j in range(FP8_BYTES):
                    t[b, p, j] = (li * 17 + tok * 3 + j * 7) & 0xFF
                # scale region: a real fp32 (power-of-2-ish), distinct per token
                scale = np.float32(2.0 ** ((tok % 13) - 6)) * np.float32(1.0 + li)
                t[b, p, SCALE_OFF:INDEXER_HIDDEN] = torch.from_numpy(
                    np.frombuffer(scale.tobytes(), dtype=np.uint8).copy()
                )
        layers[f"model.layers.{li}.self_attn.indexer.k_cache"] = t
    return layers


@torch_skip
def test_indexer_kcache_byte_and_scale_roundtrip():
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

    NL, NB, BS = 3, 16, 16
    chunk_tokens = BS * 2
    src = _make_indexer_layers(NL, NB, BS)
    names = list(src.keys())
    src_list = [src[n] for n in names]
    layer_to_group = [0] * NL

    # gather producer pages (non-contiguous) into the 1-channel MLA-style staging
    producer_pages = ((3,), (5,))
    staging = torch.zeros((1, NL, chunk_tokens, INDEXER_HIDDEN), dtype=torch.uint8)
    kv_chunk_gather(staging, src_list, producer_pages, layer_to_group, BS, use_triton=False)

    # pack exactly like the save path, unpack like the load path
    header = ChunkHeader(
        version=2,
        chunk_tokens=chunk_tokens,
        block_size=BS,
        num_layers=NL,
        layer_names=tuple(names),
        hidden_dim=INDEXER_HIDDEN,
        dtype=_torch_dtype_to_str(torch.uint8),
        cache_group_id=1000,
        num_kv_channels=1,
    )
    payload = staging.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    assert len(payload) == header.payload_bytes
    _h, got = unpack_chunk(pack_chunk_header(header) + payload)
    decoded = torch.from_numpy(np.frombuffer(got, dtype=np.uint8).copy()).reshape(
        1, NL, chunk_tokens, INDEXER_HIDDEN
    )

    # scatter into DIFFERENT physical blocks
    dst = {n: torch.zeros((NB, BS, INDEXER_HIDDEN), dtype=torch.uint8) for n in names}
    consumer_pages = ((7,), (11,))
    kv_chunk_scatter(
        decoded, [dst[n] for n in names], consumer_pages, layer_to_group, BS, use_triton=False
    )

    # byte-exact per token, and explicitly the scale region
    for n in names:
        for (spg,), (cpg,) in zip(producer_pages, consumer_pages):
            s = src[n][spg]  # [BS, 132]
            d = dst[n][cpg]
            assert torch.equal(s, d), f"{n}: indexer page {spg}->{cpg} not byte-exact"
            # scale region must be intact (the crux — no ue8m0 scale desync)
            assert torch.equal(s[:, SCALE_OFF:], d[:, SCALE_OFF:]), (
                f"{n}: ue8m0 scale region [128:132] desynced page {spg}->{cpg}"
            )
    # untouched consumer blocks stay zero
    assert torch.all(dst[names[0]][0] == 0)


MAIN_HIDDEN = 576  # MLA latent: kv_lora 512 + rope 64
_PAGE = 4096  # cuFile / hipFile file-offset alignment


@pytest.mark.parametrize(
    "chunk_tokens,indexer_dma",
    [(128, False), (256, False), (512, False), (1024, True), (2048, True), (3072, True)],
)
def test_indexer_gpu_direct_4k_alignment(chunk_tokens, indexer_dma):
    """The hipFile GPU-direct DMA fast path fires only when a chunk's PER-LAYER
    byte size is 4 KiB-aligned — cuFile requires 4 KiB file offsets and layers
    are packed back-to-back (see kvd_connector._prepare_chunk_for_prefetch_load's
    `per_layer_nbytes & (_PAGE - 1)` gate). Per-layer = chunk_tokens * hidden *
    itemsize (num_kv_channels=1 for MLA/indexer).

    The co-packed 132-byte indexer (uint8) aligns ONLY when chunk_tokens is a
    multiple of 1024 (gcd(132, 4096) = 4 -> 33*ct must divide 1024); the main
    latent (576 * 2 B) aligns at any chunk_tokens >= 32. So at the default
    INFERA_KVD_CHUNK_TOKENS 256/512 the indexer silently falls back to mmap+H2D
    while the main latent DMAs -> DSA gets only HALF the GPU-direct win.

    This is the invariant behind register_kv_caches' WARNING and the
    "set INFERA_KVD_CHUNK_TOKENS to a multiple of 1024" guidance in SKILL.md /
    KVD_DSA_INDEXER_NOTES.md. If the DMA gate or the indexer layout changes,
    update all three together.
    """
    indexer_per_layer = chunk_tokens * INDEXER_HIDDEN * 1  # uint8, kvch=1
    main_per_layer = chunk_tokens * MAIN_HIDDEN * 2  # bf16 latent
    assert (indexer_per_layer % _PAGE == 0) is indexer_dma
    assert (indexer_per_layer % _PAGE == 0) == (chunk_tokens % 1024 == 0)
    # the main latent always takes the DMA path at realistic chunk sizes
    assert main_per_layer % _PAGE == 0
