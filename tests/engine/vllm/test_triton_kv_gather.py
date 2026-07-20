###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Tests for infera/engine/vllm/triton_kv_gather.py — chunked KV
scatter / gather kernel + Python fallback. The Python fallback acts
as the correctness oracle for the Triton path; both must produce
byte-identical results on the same input.
"""

from __future__ import annotations

import importlib.util

import pytest

torch_spec = importlib.util.find_spec("torch")
torch_skip = pytest.mark.skipif(torch_spec is None, reason="torch not installed")


# Triton import is optional — if absent, the kernel module exposes a
# Python fallback only; tests that DEMAND the Triton path skip.
def _triton_available() -> bool:
    try:
        from infera.engine.vllm.triton_kv_gather import is_triton_available

        return is_triton_available()
    except ImportError:
        return False


# CUDA/ROCm device required for Triton — Triton kernels only run on
# device tensors. Python fallback runs on CPU but tests it on device
# too for parity.
def _device_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


triton_skip = pytest.mark.skipif(
    not _triton_available() or not _device_available(),
    reason="Triton or CUDA/ROCm device unavailable",
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _make_layer_tensors(num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype):
    """Build a list of `num_layers` paged-KV tensors with shape
    [2, num_blocks, block_size, num_kv_heads, head_dim]. Each layer's
    blocks are pre-filled with a deterministic per-(layer, block, kv,
    token, head, dim) value so SAVE round-trips can be verified."""
    import torch

    out = []
    for L in range(num_layers):
        t = torch.empty(
            (2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        # Fill with a unique pattern per (kv, block, token, head, dim).
        # Use a hash of indices for stability.
        for kv in range(2):
            for b in range(num_blocks):
                for tk in range(block_size):
                    val = ((L * 13 + b * 7 + kv * 3 + tk) & 0xFF) / 32.0
                    t[kv, b, tk] = val
        out.append(t)
    return out


def _make_interleaved_layer_tensors(
    num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
):
    """Build `num_layers` paged-KV tensors in vLLM 0.23's FullAttention
    INTERLEAVED layout ``[num_blocks, 2, block_size, num_kv_heads, head_dim]``
    — the size-2 K/V split sits at dim 1 (per-block), NOT dim 0. Each entry is
    filled with a unique per-(block, kv, token, head, dim) value so a gather
    can be verified against DIRECT interleaved indexing (a round-trip alone
    would not catch a symmetric stride error)."""
    import torch

    out = []
    for L in range(num_layers):
        t = torch.empty(
            (num_blocks, 2, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        for b in range(num_blocks):
            for kv in range(2):
                for tk in range(block_size):
                    val = ((L * 13 + b * 7 + kv * 3 + tk) & 0xFF) / 32.0
                    t[b, kv, tk] = val
        out.append(t)
    return out


def _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype, num_kv_channels: int = 2):
    import torch

    return torch.zeros(
        (num_kv_channels, num_layers, chunk_tokens, hidden_dim),
        dtype=dtype,
        device=device,
    )


def _make_mla_layer_tensors(num_layers, num_blocks, block_size, hidden_dim, device, dtype):
    """Build a list of `num_layers` MLA paged-KV tensors with shape
    [num_blocks, block_size, hidden_dim] (no leading 2 for K/V split —
    MLA stores a single combined kv_lora_rank + qk_rope_head_dim latent).
    Filled with a deterministic per-(layer, block, token, dim) value.
    """
    import torch

    out = []
    for L in range(num_layers):
        t = torch.empty(
            (num_blocks, block_size, hidden_dim),
            dtype=dtype,
            device=device,
        )
        for b in range(num_blocks):
            for tk in range(block_size):
                val = ((L * 13 + b * 7 + tk) & 0xFF) / 32.0
                t[b, tk] = val
        out.append(t)
    return out


def _make_mla_aiter_layer_tensors(num_layers, num_blocks, block_size, hidden_dim, device, dtype):
    """Build MLA paged-KV tensors in the ROCM_AITER_MLA physical layout:
    ``[num_blocks * block_size, 1, hidden_dim]`` — the block_size axis is
    FOLDED INTO dim 0 (so the tensor's middle dim is 1), while the engine
    still addresses block_ids with a LOGICAL block_size > 1. Measured shape
    on Kimi K2.5 is (32000, 1, 576) for 2000 blocks × block_size 16.

    The token for (logical block ``b``, offset ``off``) lives at flat row
    ``b * block_size + off``. Filled with a deterministic per-(layer, row)
    value so a gather/scatter can be checked against direct flat indexing.
    """
    import torch

    out = []
    total_rows = num_blocks * block_size
    for L in range(num_layers):
        t = torch.empty((total_rows, 1, hidden_dim), dtype=dtype, device=device)
        for r in range(total_rows):
            val = ((L * 13 + r * 7) & 0xFF) / 32.0
            t[r, 0] = val
        out.append(t)
    return out


# ----------------------------------------------------------------------
# Python fallback — round-trip sanity
# ----------------------------------------------------------------------


@torch_skip
def test_python_fallback_round_trip_on_cpu_no_hma():
    """Python loop: gather (paged → staging), zero out paged, scatter
    back (staging → paged), verify bytes restored. Single cache_group
    (non-HMA). CPU-only — Triton path is forced off by passing
    use_triton=False so this is pure correctness oracle."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 8, 16
    num_kv_heads, head_dim = 2, 8
    chunk_tokens = block_size * 4  # 4 pages per chunk
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cpu")
    dtype = torch.float32

    layers = _make_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    # Page → block id: page 0 → block 2, page 1 → block 3, etc.
    per_page_block_ids = ((2,), (3,), (5,), (7,))
    layer_to_group = [0] * num_layers

    # Snapshot the relevant slots BEFORE gather.
    expected = []
    for L in range(num_layers):
        for page_idx, page_ids in enumerate(per_page_block_ids):
            expected.append(layers[L][:, page_ids[0]].clone())

    # Gather paged → staging.
    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    # Zero the paged slots we'll restore — verify scatter actually rewrites them.
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers[L][:, page_ids[0]].zero_()

    # Scatter staging → paged.
    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    # Verify each paged slot matches its pre-gather snapshot.
    i = 0
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            got = layers[L][:, page_ids[0]]
            assert torch.equal(got, expected[i]), f"layer {L} page {page_ids} mismatch"
            i += 1


@torch_skip
def test_python_fallback_mla_round_trip_on_cpu():
    """MLA tensor layout (single combined KV latent, no K/V split):
    paged shape is [num_blocks, block_size, hidden_dim], staging is
    [1, num_layers, chunk_tokens, hidden_dim]. Round-trip must
    reproduce bytes exactly.
    """
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 8, 16
    hidden_dim = 576  # kv_lora_rank (512) + qk_rope_head_dim (64), DeepseekV3 MLA shape
    chunk_tokens = block_size * 4
    device = torch.device("cpu")
    dtype = torch.float32

    layers = _make_mla_layer_tensors(num_layers, num_blocks, block_size, hidden_dim, device, dtype)
    per_page_block_ids = ((2,), (3,), (5,), (7,))
    layer_to_group = [0] * num_layers

    expected = []
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            expected.append(layers[L][page_ids[0]].clone())

    # MLA staging has leading dim 1.
    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype, num_kv_channels=1)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    # Zero out the slots we'll restore.
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers[L][page_ids[0]].zero_()

    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    i = 0
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            got = layers[L][page_ids[0]]
            assert torch.equal(got, expected[i]), f"MLA layer {L} page {page_ids} mismatch"
            i += 1


@torch_skip
def test_python_fallback_hma_two_groups():
    """HMA: layers 0,1 → group 0 (full-attn block ids); layers 2,3 →
    group 1 (sliding block ids). Verify each layer uses the right
    per-group block_id mapping (not just group 0's)."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 16, 16
    num_kv_heads, head_dim = 2, 4
    chunk_tokens = block_size * 2  # 2 pages
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cpu")
    dtype = torch.float32

    layers = _make_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    # Page 0: group 0 → block 3, group 1 → block 7
    # Page 1: group 0 → block 4, group 1 → block 9
    per_page_block_ids = ((3, 7), (4, 9))
    layer_to_group = [0, 0, 1, 1]

    # Snapshot the right blocks per layer's group.
    snapshots = {}
    for L in range(num_layers):
        gid = layer_to_group[L]
        for page_idx, page_ids in enumerate(per_page_block_ids):
            bid = page_ids[gid]
            snapshots[(L, page_idx)] = layers[L][:, bid].clone()

    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    # Zero the targeted slots, then scatter back.
    for L in range(num_layers):
        gid = layer_to_group[L]
        for page_ids in per_page_block_ids:
            layers[L][:, page_ids[gid]].zero_()

    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    for L in range(num_layers):
        gid = layer_to_group[L]
        for page_idx, page_ids in enumerate(per_page_block_ids):
            bid = page_ids[gid]
            assert torch.equal(layers[L][:, bid], snapshots[(L, page_idx)]), (
                f"HMA mismatch layer {L} group {gid} block {bid}"
            )


# ----------------------------------------------------------------------
# Regular attention — INTERLEAVED layout [num_blocks, 2, block_size, H]
# (vLLM 0.23 FullAttention). The K/V split axis is dim 1, so a block
# occupies 2*block_size*hidden_dim contiguous elements (K then V). The old
# code only knew the OUTERMOST [2, num_blocks, ...] layout and dropped /
# mis-strided this one, silently scrambling KV on reload. NOTE: a plain
# gather→scatter round-trip is INSUFFICIENT here — a symmetric stride error
# writes and reads the same wrong place and still round-trips. So the gather
# oracle below asserts against DIRECT interleaved indexing.
# ----------------------------------------------------------------------


@torch_skip
def test_python_fallback_interleaved_gather_matches_direct_indexing_cpu():
    """Gather from the interleaved [B, 2, P, H] layout must pull each token's
    K and V from `layer[b, kv, off]` — verified against direct indexing, which
    (unlike a round-trip) catches a symmetric stride error."""
    import torch

    from infera.engine.vllm.triton_kv_gather import kv_chunk_gather

    num_layers, num_blocks, block_size = 3, 8, 16
    num_kv_heads, head_dim = 2, 8
    chunk_tokens = block_size * 4
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cpu")
    dtype = torch.float32

    layers = _make_interleaved_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    # Non-contiguous, non-trivial block ids (b>0 exercises the per-block gap).
    per_page_block_ids = ((2,), (3,), (5,), (7,))
    layer_to_group = [0] * num_layers

    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    for L in range(num_layers):
        for page_idx, page_ids in enumerate(per_page_block_ids):
            b = page_ids[0]
            for off in range(block_size):
                for kv in range(2):
                    got = staging[kv, L, page_idx * block_size + off, :]
                    exp = layers[L][b, kv, off].reshape(hidden_dim)
                    assert torch.equal(got, exp), (
                        f"interleaved gather wrong row: layer {L} block {b} off {off} kv {kv}"
                    )


@torch_skip
def test_python_fallback_interleaved_round_trip_cpu():
    """Full round-trip on the interleaved layout: gather → zero the touched
    blocks → scatter → bytes restored exactly."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 8, 16
    num_kv_heads, head_dim = 2, 8
    chunk_tokens = block_size * 4
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cpu")
    dtype = torch.float32

    layers = _make_interleaved_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    per_page_block_ids = ((2,), (3,), (5,), (7,))
    layer_to_group = [0] * num_layers

    expected = []
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            expected.append(layers[L][page_ids[0]].clone())

    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers[L][page_ids[0]].zero_()
    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    i = 0
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            assert torch.equal(layers[L][page_ids[0]], expected[i]), (
                f"interleaved round-trip mismatch: layer {L} page {page_ids}"
            )
            i += 1


@torch_skip
def test_interleaved_and_outermost_are_distinguished_not_aliased():
    """Guard against a regression that treats the two regular-attention
    layouts as the same. Fill an interleaved [B,2,P,H] tensor and an outermost
    [2,B,P,H] tensor with the SAME logical (block,kv,token) values, gather both
    for the same block id > 0, and require the two staging results to be EQUAL
    (each read its own layout correctly). A stride error that ignored the
    per-block gap would gather the wrong rows from the interleaved tensor and
    they would differ."""
    import torch

    from infera.engine.vllm.triton_kv_gather import kv_chunk_gather

    num_layers, num_blocks, block_size = 2, 8, 16
    num_kv_heads, head_dim = 2, 8
    chunk_tokens = block_size * 2
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cpu")
    dtype = torch.float32

    inter = _make_interleaved_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    # Build the outermost twin with identical logical values: outer[kv,b,tk] == inter[b,kv,tk].
    outer = _make_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    per_page_block_ids = ((3,), (6,))
    layer_to_group = [0] * num_layers

    s_inter = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    s_outer = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        s_inter, inter, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_gather(
        s_outer, outer, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    assert torch.equal(s_inter, s_outer), (
        "interleaved and outermost layouts gathered different bytes for the "
        "same logical KV — one of the layout paths is mis-strided"
    )


@triton_skip
def test_triton_matches_python_interleaved_layout():
    """Triton path must match the Python oracle on the interleaved
    [B, 2, P, H] layout for both gather and scatter."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 16, 16
    num_kv_heads, head_dim = 4, 16
    chunk_tokens = block_size * 4
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    layers_py = _make_interleaved_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    layers_tt = [t.clone() for t in layers_py]
    per_page_block_ids = ((2,), (5,), (8,), (11,))
    layer_to_group = [0] * num_layers

    staging_py = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    staging_tt = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging_py, layers_py, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_gather(
        staging_tt,
        layers_tt,
        per_page_block_ids,
        layer_to_group,
        block_size,
        use_triton=True,
        sync=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(staging_py, staging_tt), (
        "Triton gather diverged from Python oracle (interleaved)"
    )

    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers_py[L][page_ids[0]].zero_()
            layers_tt[L][page_ids[0]].zero_()
    kv_chunk_scatter(
        staging_py, layers_py, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_scatter(
        staging_tt,
        layers_tt,
        per_page_block_ids,
        layer_to_group,
        block_size,
        use_triton=True,
        sync=True,
    )
    torch.cuda.synchronize()
    for L in range(num_layers):
        assert torch.equal(layers_py[L], layers_tt[L]), (
            f"Triton scatter diverged (interleaved) layer {L}"
        )


# ----------------------------------------------------------------------
# Triton-equals-Python oracle (requires device + Triton)
# ----------------------------------------------------------------------


@triton_skip
def test_triton_matches_python_no_hma():
    """For the same input the Triton path must produce byte-identical
    output to the Python fallback. Run both on device, compare."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 8, 16, 16
    num_kv_heads, head_dim = 4, 16
    chunk_tokens = block_size * 4
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    layers_py = _make_layer_tensors(
        num_layers, num_blocks, block_size, num_kv_heads, head_dim, device, dtype
    )
    layers_tt = [t.clone() for t in layers_py]
    per_page_block_ids = ((2,), (5,), (8,), (11,))
    layer_to_group = [0] * num_layers

    # Save (paged → staging) with both paths; compare staging tensors.
    staging_py = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    staging_tt = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging_py, layers_py, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_gather(
        staging_tt, layers_tt, per_page_block_ids, layer_to_group, block_size, use_triton=True
    )
    torch.cuda.synchronize()
    assert torch.equal(staging_py, staging_tt), "Triton gather diverged from Python fallback"

    # Load (staging → paged) with both — zero out paged first.
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers_py[L][:, page_ids[0]].zero_()
            layers_tt[L][:, page_ids[0]].zero_()
    kv_chunk_scatter(
        staging_py, layers_py, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_scatter(
        staging_tt, layers_tt, per_page_block_ids, layer_to_group, block_size, use_triton=True
    )
    torch.cuda.synchronize()
    for L in range(num_layers):
        assert torch.equal(layers_py[L], layers_tt[L]), f"Triton scatter diverged on layer {L}"


@triton_skip
def test_triton_round_trip_perf_smoke():
    """End-to-end round-trip on device with realistic shapes
    (gpt-oss-120b-ish: 36 layers × 64 tokens × 64 hidden). Just a
    smoke check that the kernel runs without errors on a production-
    shaped chunk; no perf assertion."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 36, 64, 64
    num_kv_heads, head_dim = 1, 64  # TP=8 worth
    chunk_tokens = 512
    hidden_dim = num_kv_heads * head_dim
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    layers = [
        torch.randn(
            (2, num_blocks, block_size, num_kv_heads, head_dim),
            dtype=dtype,
            device=device,
        )
        for _ in range(num_layers)
    ]
    per_page_block_ids = tuple((i + 3,) for i in range(chunk_tokens // block_size))
    layer_to_group = [0] * num_layers

    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=True
    )
    torch.cuda.synchronize()
    # Zero and scatter back.
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers[L][:, page_ids[0]].zero_()
    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=True
    )
    torch.cuda.synchronize()
    # No assertion on perf; just that it ran.


# ----------------------------------------------------------------------
# Edge cases
# ----------------------------------------------------------------------


@triton_skip
def test_triton_mla_round_trip_on_cuda():
    """MLA round-trip on the actual Triton path (mirror of
    `test_python_fallback_mla_round_trip_on_cpu` but on a real CUDA
    device). Catches drift between the Python oracle (which we trust by
    inspection) and the Triton scatter/gather kernel for the
    single-channel (num_kv_channels=1) MLA layout. Adds the missing
    rung in our correctness ladder: CPU-python passes; CUDA-Triton
    must also pass under the same fill pattern.
    """
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 8, 16
    hidden_dim = 576  # kv_lora_rank (512) + qk_rope_head_dim (64), DeepseekV3 MLA shape
    chunk_tokens = block_size * 4
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    layers = _make_mla_layer_tensors(num_layers, num_blocks, block_size, hidden_dim, device, dtype)
    per_page_block_ids = ((2,), (3,), (5,), (7,))
    layer_to_group = [0] * num_layers

    expected = []
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            expected.append(layers[L][page_ids[0]].clone())

    # MLA staging has leading dim 1; mirror the CPU oracle test.
    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype, num_kv_channels=1)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=True
    )
    torch.cuda.synchronize()

    # Zero out the slots we'll restore, then scatter back.
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            layers[L][page_ids[0]].zero_()

    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=True
    )
    torch.cuda.synchronize()

    i = 0
    for L in range(num_layers):
        for page_ids in per_page_block_ids:
            got = layers[L][page_ids[0]]
            assert torch.equal(got, expected[i]), (
                f"MLA Triton round-trip mismatch: layer {L} page {page_ids}"
            )
            i += 1


# ----------------------------------------------------------------------
# MLA — ROCM_AITER_MLA flat layout [num_blocks*block_size, 1, hidden]
# (logical block_size != tensor middle dim). This is the layout that
# caused kvd L3 ext_hit=0% on Kimi K2.5: deriving block_size from the
# tensor shape gave 1, desyncing the chunk grain and (had saves landed)
# mis-indexing the scatter. The kernel must address slot = bid*block_size
# + offset over the flat buffer using the LOGICAL block_size.
# ----------------------------------------------------------------------


@torch_skip
def test_python_fallback_mla_aiter_flat_layout_round_trip_on_cpu():
    """AITER flat MLA layout [num_blocks*block_size, 1, hidden] with a
    LOGICAL block_size of 16 (≠ the tensor's middle dim of 1). Gather must
    pull the right flat rows; round-trip must restore bytes exactly.
    """
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 3, 8, 16
    hidden_dim = 576  # DeepseekV3/Kimi MLA: kv_lora_rank 512 + qk_rope 64
    chunk_tokens = block_size * 4  # 4 logical pages per chunk
    device = torch.device("cpu")
    dtype = torch.float32

    layers = _make_mla_aiter_layer_tensors(
        num_layers, num_blocks, block_size, hidden_dim, device, dtype
    )
    # Logical block ids — NOT contiguous, to catch slot-math errors.
    per_page_block_ids = ((2,), (3,), (5,), (7,))
    layer_to_group = [0] * num_layers

    # Direct-flat-indexing oracle: token (page p=logical block b, off) lives
    # at flat row b*block_size+off.
    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, device, dtype, num_kv_channels=1)
    kv_chunk_gather(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )

    for L in range(num_layers):
        flat = layers[L].view(-1, hidden_dim)
        for page_idx, page_ids in enumerate(per_page_block_ids):
            b = page_ids[0]
            for off in range(block_size):
                got = staging[0, L, page_idx * block_size + off, :]
                exp = flat[b * block_size + off, :]
                assert torch.equal(got, exp), f"gather wrong row: layer {L} block {b} off {off}"

    # Round-trip: snapshot, zero the touched flat rows, scatter back.
    expected = [layers[L].clone() for L in range(num_layers)]
    for L in range(num_layers):
        flat = layers[L].view(-1, hidden_dim)
        for page_ids in per_page_block_ids:
            b = page_ids[0]
            flat[b * block_size : (b + 1) * block_size].zero_()
    kv_chunk_scatter(
        staging, layers, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    for L in range(num_layers):
        assert torch.equal(layers[L], expected[L]), f"round-trip mismatch on layer {L}"


@torch_skip
def test_python_fallback_mla_flat_layout_block_size_one_is_subset():
    """Sanity that block_size=1 (the WRONG value the old code derived from
    the AITER tensor's shape[1]) only touches 1 row per page — i.e. it
    would have transferred 1/block_size of the KV. Documents why deriving
    block_size from the tensor shape was incorrect, not just suboptimal.
    """
    import torch

    from infera.engine.vllm.triton_kv_gather import kv_chunk_gather

    num_layers, num_blocks = 1, 8
    hidden_dim = 32
    logical_block_size = 16
    layers = _make_mla_aiter_layer_tensors(
        num_layers, num_blocks, logical_block_size, hidden_dim, torch.device("cpu"), torch.float32
    )
    per_page_block_ids = ((2,), (3,))
    # With block_size=1, chunk_tokens must be pages*1 = 2.
    staging = _empty_staging(
        num_layers, 2, hidden_dim, torch.device("cpu"), torch.float32, num_kv_channels=1
    )
    kv_chunk_gather(staging, layers, per_page_block_ids, [0], block_size=1, use_triton=False)
    flat = layers[0].view(-1, hidden_dim)
    # block_size=1 → page p (block b) gathers ONLY flat row b*1 = b, NOT the
    # block's true rows b*16..b*16+15. So it grabs the wrong rows entirely.
    assert torch.equal(staging[0, 0, 0, :], flat[2, :])  # block 2 → row 2 (wrong: should be row 32)
    assert torch.equal(staging[0, 0, 1, :], flat[3, :])


@triton_skip
def test_triton_matches_python_mla_aiter_flat_layout():
    """Triton path must match the Python oracle on the AITER flat MLA
    layout for both gather and scatter, using the logical block_size."""
    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    num_layers, num_blocks, block_size = 4, 16, 16
    hidden_dim = 576
    chunk_tokens = block_size * 4
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    layers_py = _make_mla_aiter_layer_tensors(
        num_layers, num_blocks, block_size, hidden_dim, device, dtype
    )
    layers_tt = [t.clone() for t in layers_py]
    per_page_block_ids = ((2,), (5,), (8,), (11,))
    layer_to_group = [0] * num_layers

    staging_py = _empty_staging(
        num_layers, chunk_tokens, hidden_dim, device, dtype, num_kv_channels=1
    )
    staging_tt = _empty_staging(
        num_layers, chunk_tokens, hidden_dim, device, dtype, num_kv_channels=1
    )
    kv_chunk_gather(
        staging_py, layers_py, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_gather(
        staging_tt,
        layers_tt,
        per_page_block_ids,
        layer_to_group,
        block_size,
        use_triton=True,
        sync=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(staging_py, staging_tt), "Triton gather diverged on AITER flat MLA layout"

    for L in range(num_layers):
        flat_py = layers_py[L].view(-1, hidden_dim)
        flat_tt = layers_tt[L].view(-1, hidden_dim)
        for page_ids in per_page_block_ids:
            b = page_ids[0]
            flat_py[b * block_size : (b + 1) * block_size].zero_()
            flat_tt[b * block_size : (b + 1) * block_size].zero_()
    kv_chunk_scatter(
        staging_py, layers_py, per_page_block_ids, layer_to_group, block_size, use_triton=False
    )
    kv_chunk_scatter(
        staging_tt,
        layers_tt,
        per_page_block_ids,
        layer_to_group,
        block_size,
        use_triton=True,
        sync=True,
    )
    torch.cuda.synchronize()
    for L in range(num_layers):
        assert torch.equal(layers_py[L], layers_tt[L]), (
            f"Triton scatter diverged on AITER flat MLA layer {L}"
        )


@triton_skip
def test_triton_mla_bounds_guard_raises_on_out_of_range_block_id():
    """A logical block_id whose slot (bid*block_size) exceeds the layer's
    row capacity must RAISE (caller degrades to a cache miss) rather than
    silently faulting the GPU. Guards the OOB-scatter crash class."""
    import torch

    from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

    num_blocks, block_size, hidden_dim = 8, 16, 64  # capacity = 128 rows
    layers = _make_mla_aiter_layer_tensors(
        1, num_blocks, block_size, hidden_dim, torch.device("cuda:0"), torch.bfloat16
    )
    staging = _empty_staging(
        1, block_size, hidden_dim, torch.device("cuda:0"), torch.bfloat16, num_kv_channels=1
    )
    # block id 99 → slot 99*16=1584 >> 128 rows.
    with pytest.raises((ValueError, AssertionError)):
        kv_chunk_scatter(
            staging, layers, ((99,),), [0], block_size=block_size, use_triton=True, sync=True
        )


# ----------------------------------------------------------------------
# Cross-device executor thread (the TP>1 / DP-attention hazard). Under
# TP>1 the engine pins each worker process to cuda:rank on its MAIN
# thread, but kvd's load runs in _load_executor THREADS whose torch
# current-device defaults to 0. The connector fixes this by calling
# torch.cuda.set_device(layer.device) before the scatter; this test
# guards that the kernel is correct when invoked from such a thread on a
# NON-zero device with the device pinned (mirrors the connector fix).
# ----------------------------------------------------------------------


_multi_gpu_skip = pytest.mark.skipif(
    not _triton_available()
    or not _device_available()
    or (_device_available() and __import__("torch").cuda.device_count() < 2),
    reason="needs Triton + >=2 CUDA/ROCm devices",
)


@_multi_gpu_skip
def test_scatter_gather_from_executor_thread_on_nonzero_device():
    """MLA scatter/gather invoked from a ThreadPoolExecutor worker thread
    (default device 0) against KV tensors on cuda:1, with the device
    pinned inside the worker (as InferaKvdConnector._load_chunk_packed
    does). Reproduces the TP>1 / DP-attention per-rank condition in one
    process; must round-trip correctly with no GPU fault.
    """
    import concurrent.futures

    import torch

    from infera.engine.vllm.triton_kv_gather import (
        kv_chunk_gather,
        kv_chunk_scatter,
    )

    dev = torch.device("cuda:1")
    num_layers, num_blocks, block_size = 4, 16, 16
    hidden_dim = 576
    chunk_tokens = block_size * 4
    dtype = torch.bfloat16

    layers = _make_mla_aiter_layer_tensors(
        num_layers, num_blocks, block_size, hidden_dim, dev, dtype
    )
    per_page_block_ids = ((2,), (5,), (8,), (11,))
    layer_to_group = [0] * num_layers
    expected = [t.clone() for t in layers]

    staging = _empty_staging(num_layers, chunk_tokens, hidden_dim, dev, dtype, num_kv_channels=1)

    def _worker():
        # An executor thread's torch current-device defaults to 0, NOT the
        # tensors' device — exactly the TP>1 hazard. The connector pins it;
        # mirror that here. Without this line the kernel would launch on
        # cuda:0 against cuda:1 pointers → GPU memory fault.
        assert torch.cuda.current_device() == 0, "fresh thread should default to device 0"
        torch.cuda.set_device(dev)
        kv_chunk_gather(
            staging,
            layers,
            per_page_block_ids,
            layer_to_group,
            block_size,
            use_triton=True,
            sync=True,
        )
        for L in range(num_layers):
            flat = layers[L].view(-1, hidden_dim)
            for page_ids in per_page_block_ids:
                b = page_ids[0]
                flat[b * block_size : (b + 1) * block_size].zero_()
        kv_chunk_scatter(
            staging,
            layers,
            per_page_block_ids,
            layer_to_group,
            block_size,
            use_triton=True,
            sync=True,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        ex.submit(_worker).result()
    torch.cuda.synchronize(dev)

    for L in range(num_layers):
        assert torch.equal(layers[L], expected[L]), (
            f"cross-device executor-thread round-trip mismatch on layer {L}"
        )


@torch_skip
def test_python_fallback_chunk_tokens_must_be_multiple_of_block_size():
    """Assert chunk_tokens % block_size == 0 invariant."""
    import torch

    from infera.engine.vllm.triton_kv_gather import kv_chunk_scatter

    staging = torch.zeros((2, 1, 17, 4), dtype=torch.float32)  # 17 tokens, block 16 → not divisible
    layers = [torch.zeros((2, 4, 16, 1, 4), dtype=torch.float32)]
    with pytest.raises(AssertionError):
        kv_chunk_scatter(staging, layers, ((0,),), [0], block_size=16, use_triton=False)
