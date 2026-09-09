"""Sampler vocabulary softmax — the kernel under optimization.

Traced hot kernel from an sglang v0.5.14 decode step on AMD MI300X (gfx942),
model Qwen/Qwen3-0.6B. sglang's sampler does, per decode step:

    # python/sglang/srt/layers/sampler.py:183
    logits[:] = torch.softmax(logits, dim=-1)

with ``logits`` of shape ``[running_requests, vocab_size]`` in float32. With 8
concurrent requests that is ``[8, 151936]`` fp32, and ATen dispatches
``cunn_SoftMaxForwardGmem<4, float, float, float, ...>`` — the global-memory
fallback used when the reduced dimension is too large for the shared-memory
path.

Measured baseline (torch.profiler, DECODE stage, batch 8):
    55.59 us/call, 14.50% of all decode GPU time.
    Traffic 8*151936*4 B read + same written = 9.72 MB -> ~175 GB/s achieved,
    against ~5.3 TB/s HBM peak on MI300X. Memory-bound with large headroom.

Implementation: with only batch=1/8/32 rows, one-workgroup-per-row caps CU
occupancy at <=32 of MI300X's 304 CUs. This implementation splits each row
into ``segments_per_row`` contiguous chunks (sized so total workgroups land
around ~1024) and runs a 3-kernel segmented online-softmax:

  1. stats:      grid (batch, segments) -> per-segment (local_max, local_sum)
  2. combine:    grid (batch,)          -> merge segment stats -> (row_max, row_sum)
  3. normalize:  grid (batch, segments) -> exp(x - row_max) / row_sum -> out

This reads logits twice (stats + normalize) instead of once (50% more bytes
moved than a single-pass kernel would), but fills far more CUs in parallel,
which dominates at these batch sizes.

Note: an earlier iteration tried folding _combine_kernel's reduction directly
into _normalize_kernel (each (row, segment) program redundantly re-reducing
the row's pmax/psum arrays) to cut launch count 3->2. That regressed at low
batch: the redundant-reduction cost scales as batch * segments *
next_pow2(segments), which is ~8MB of extra traffic at B1 (segments=1024) --
comparable to the whole row's data volume -- more than it saved in launch
overhead. Kept as a separate tiny _combine_kernel instead.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch

_TARGET_WORKGROUPS = 1216
_BLOCK_CAP = 2048
# Fold _combine_kernel's reduction into _normalize_kernel (2-launch pipeline)
# only when `segments` is small enough that the O(segments) redundant
# per-program reduction stays cheap relative to the launch it removes. At
# large `segments` (e.g. B1's 1216) the redundant reduction costs ~O(segments^2)
# total reads and swamps real vocab traffic; at small `segments` (B8's 152,
# B32's 38) it is negligible and removing the combine launch's fixed tax wins.
_FOLD_THRESHOLD = 256


def _segments_per_row(batch: int) -> int:
    return max(1, -(-_TARGET_WORKGROUPS // batch))


@triton.jit
def _stats_kernel(
    x_ptr,
    pmax_ptr,
    psum_ptr,
    vocab,
    seg_size,
    stride_row,
    segments,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seg = tl.program_id(1)

    start = seg * seg_size
    seg_end = min(start + seg_size, vocab)
    row_ptr = x_ptr + row * stride_row

    # Online (streaming) max+sum over the segment in BLOCK-sized chunks. A
    # capped BLOCK keeps per-program register footprint bounded regardless of
    # seg_size, so occupancy stays high even when segments are large.
    local_max = -float("inf")
    local_sum = 0.0
    for base in range(start, seg_end, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < seg_end
        x = tl.load(row_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)
        chunk_max = tl.max(x, axis=0)
        new_max = tl.maximum(local_max, chunk_max)
        shifted = tl.where(mask, x - new_max, -float("inf"))
        chunk_sum = tl.sum(tl.exp(shifted), axis=0)
        local_sum = local_sum * tl.exp(local_max - new_max) + chunk_sum
        local_max = new_max

    out_idx = row * segments + seg
    tl.store(pmax_ptr + out_idx, local_max)
    tl.store(psum_ptr + out_idx, local_sum)


@triton.jit
def _combine_kernel(
    pmax_ptr,
    psum_ptr,
    row_max_ptr,
    row_sum_ptr,
    segments,
    BLOCK_SEG: tl.constexpr,
):
    row = tl.program_id(0)
    idx = tl.arange(0, BLOCK_SEG)
    mask = idx < segments

    base = row * segments
    pm = tl.load(pmax_ptr + base + idx, mask=mask, other=-float("inf"))
    ps = tl.load(psum_ptr + base + idx, mask=mask, other=0.0)

    row_max = tl.max(pm, axis=0)
    scale = tl.exp(pm - row_max)
    row_sum = tl.sum(ps * scale, axis=0)

    tl.store(row_max_ptr + row, row_max)
    tl.store(row_sum_ptr + row, row_sum)


@triton.jit
def _normalize_kernel(
    x_ptr,
    out_ptr,
    row_max_ptr,
    row_sum_ptr,
    vocab,
    seg_size,
    stride_row,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seg = tl.program_id(1)

    start = seg * seg_size
    seg_end = min(start + seg_size, vocab)

    row_ptr = x_ptr + row * stride_row
    out_row_ptr = out_ptr + row * stride_row

    row_max = tl.load(row_max_ptr + row)
    inv_sum = 1.0 / tl.load(row_sum_ptr + row)

    for base in range(start, seg_end, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < seg_end
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - row_max) * inv_sum
        tl.store(out_row_ptr + offs, y, mask=mask)


@triton.jit
def _normalize_kernel_fused(
    x_ptr,
    out_ptr,
    pmax_ptr,
    psum_ptr,
    vocab,
    seg_size,
    stride_row,
    segments,
    BLOCK: tl.constexpr,
    BLOCK_SEG: tl.constexpr,
):
    """2-launch variant: each (row, segment) program redundantly reduces the
    row's per-segment stats itself (no separate _combine_kernel launch),
    then normalizes its own segment. Only used when `segments` is small
    enough that the O(segments) redundant reduction per program is cheap
    relative to the launch it removes (see module docstring / gating in
    `sampler_softmax`)."""
    row = tl.program_id(0)
    seg = tl.program_id(1)

    idx = tl.arange(0, BLOCK_SEG)
    seg_mask = idx < segments
    base_stats = row * segments
    pm = tl.load(pmax_ptr + base_stats + idx, mask=seg_mask, other=-float("inf"))
    ps = tl.load(psum_ptr + base_stats + idx, mask=seg_mask, other=0.0)

    row_max = tl.max(pm, axis=0)
    scale = tl.exp(pm - row_max)
    row_sum = tl.sum(ps * scale, axis=0)
    inv_sum = 1.0 / row_sum

    start = seg * seg_size
    seg_end = min(start + seg_size, vocab)

    row_ptr = x_ptr + row * stride_row
    out_row_ptr = out_ptr + row * stride_row

    for base in range(start, seg_end, BLOCK):
        offs = base + tl.arange(0, BLOCK)
        mask = offs < seg_end
        x = tl.load(row_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - row_max) * inv_sum
        tl.store(out_row_ptr + offs, y, mask=mask)


_stats_cache: dict[int, torch.Tensor] = {}


def _get_scratch(batch: int, segments: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    key = (batch, segments, device, dtype)
    cache = _get_scratch._cache
    buf = cache.get(key)
    if buf is None:
        pmax = torch.empty((batch, segments), device=device, dtype=torch.float32)
        psum = torch.empty((batch, segments), device=device, dtype=torch.float32)
        row_max = torch.empty((batch,), device=device, dtype=torch.float32)
        row_sum = torch.empty((batch,), device=device, dtype=torch.float32)
        buf = (pmax, psum, row_max, row_sum)
        cache[key] = buf
    return buf


_get_scratch._cache = {}


def sampler_softmax(logits: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the vocabulary dimension.

    Args:
        logits: ``[batch, vocab]`` float32 tensor of raw logits.
        out:    ``[batch, vocab]`` float32 tensor receiving the probabilities.
                Pre-allocated by the caller so the benchmark can be graph-captured.

    Returns:
        ``out``.
    """
    assert logits.is_cuda and logits.dtype == torch.float32
    batch, vocab = logits.shape
    stride_row = logits.stride(0)
    assert logits.stride(1) == 1

    segments = _segments_per_row(batch)
    # Round seg_size up to a multiple of 4 elements (16B) so segment starts
    # stay dwordx4-aligned for coalesced loads/stores; masking already handles
    # the uneven final segment via min(start+seg_size, vocab).
    seg_size = -(-vocab // segments)
    seg_size = (seg_size + 3) & ~3
    BLOCK = min(triton.next_power_of_2(seg_size), _BLOCK_CAP)
    BLOCK_SEG = triton.next_power_of_2(segments)

    pmax, psum, row_max, row_sum = _get_scratch(batch, segments, logits.device, logits.dtype)

    num_warps = 4 if BLOCK >= 512 else 2

    _stats_kernel[(batch, segments)](
        logits,
        pmax,
        psum,
        vocab,
        seg_size,
        stride_row,
        segments,
        BLOCK=BLOCK,
        num_warps=num_warps,
    )

    if segments <= _FOLD_THRESHOLD:
        # 2-launch path: normalize does the row-stats reduction itself.
        _normalize_kernel_fused[(batch, segments)](
            logits,
            out,
            pmax,
            psum,
            vocab,
            seg_size,
            stride_row,
            segments,
            BLOCK=BLOCK,
            BLOCK_SEG=BLOCK_SEG,
            num_warps=num_warps,
        )
    else:
        _combine_kernel[(batch,)](
            pmax,
            psum,
            row_max,
            row_sum,
            segments,
            BLOCK_SEG=BLOCK_SEG,
            num_warps=2,
        )

        _normalize_kernel[(batch, segments)](
            logits,
            out,
            row_max,
            row_sum,
            vocab,
            seg_size,
            stride_row,
            BLOCK=BLOCK,
            num_warps=num_warps,
        )

    return out
