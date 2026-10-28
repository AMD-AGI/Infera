"""GLM-5.2 DCP4 decode indexer, one DCP rank (rank 0 of 4), as sglang 93034eba91 runs it on gfx950.

ONE public entry point: ``dcp_indexer_topk`` = Indexer._dcp_split_decode_topk -> split_decode_topk
(srt/layers/attention/dsa/dsa_indexer.py:1582, dsa_dcp_split_indexer.py:150) for TARGET_VERIFY rows with
SGLANG_DSA_DCP_SPLIT_DECODE_INDEXER=1 + SGLANG_DSA_DCP_SPLIT_INDEXER_TRITON=1, SGLANG_DSA_FUSE_TOPK=1, sgl-kernel top-k:
  1. logits over the pages this rank owns (pages r, r+4, ...; the index K cache is replicated on every rank):
       aiter_paged_mqa_logits(q, kv, weights, owned_lens, owned_tables, owned_pages*64, preshuffle, 64)
  2. pad the logits to >= topk columns with -inf
  3. local top-2048 per row: sgl_kernel.fast_topk_v2(logits, owned_lens, 2048)
  4. pack (score bits, global position) [2, rows, 2048]: pack_candidates_triton
  5. all_gather over the 4 DCP ranks (RCCL; modelled here by concatenating the other ranks' packs, given as input)
  6. merge the 4 x 2048 candidates to the global top-2048 and map positions to KV slots: merge_candidates_triton
Steps 1/3 call the production aiter / sgl-kernel ops; 4/6 are verbatim copies of
dsa_dcp_split_indexer_triton.py (93034eba91). Everything may be replaced by in-file kernels.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

PAGE_SIZE = 64
DCP = 4
RANK = 0


@triton.jit
def _pack_candidates_kernel(
    logits_ptr,
    logits_stride,
    width,
    idx_ptr,
    idx_stride,
    lens_ptr,
    out_ptr,
    rows,
    TOPK: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    W: tl.constexpr,
    R: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    lane = offs < TOPK
    idx = tl.load(idx_ptr + row * idx_stride + offs, mask=lane, other=-1).to(tl.int64)
    length = tl.load(lens_ptr + row).to(tl.int64)
    valid = lane & (idx >= 0) & (idx < length) & (idx < width)
    safe = tl.where(valid, idx, 0)
    score = tl.load(
        logits_ptr + row * logits_stride + safe, mask=valid, other=float("-inf")
    ).to(tl.float32)
    score = tl.where(valid, score, float("-inf"))
    pos = ((safe // PAGE) * W + R) * PAGE + safe % PAGE
    pos = tl.where(valid, pos, -1).to(tl.int32)
    # out is [2, rows, TOPK] int32: fp32 score bits, then the request position.
    tl.store(out_ptr + row * TOPK + offs, score.to(tl.int32, bitcast=True), mask=lane)
    tl.store(out_ptr + (rows + row) * TOPK + offs, pos, mask=lane)


def pack_candidates_triton(
    local_logits: torch.Tensor,
    local_idx: torch.Tensor,
    local_lens: torch.Tensor,
    topk: int,
    page: int,
    w: int,
    r: int,
) -> torch.Tensor:
    """Same contract as ``dsa_dcp_split_indexer.pack_candidates``."""
    rows = local_idx.shape[0]
    out = torch.empty((2, rows, topk), dtype=torch.int32, device=local_idx.device)
    if rows == 0:
        return out
    assert local_logits.stride(1) == 1 and local_idx.stride(1) == 1
    _pack_candidates_kernel[(rows,)](
        local_logits,
        local_logits.stride(0),
        local_logits.shape[1],
        local_idx,
        local_idx.stride(0),
        local_lens,
        out,
        rows,
        TOPK=topk,
        BLOCK=triton.next_power_of_2(topk),
        PAGE=page,
        W=w,
        R=r,
    )
    return out


@triton.jit
def _merge_candidates_kernel(
    packs_ptr,
    rows,
    page_table_ptr,
    page_table_stride,
    out_ptr,
    TOPK: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGE: tl.constexpr,
    HAS_PAGE_TABLE: tl.constexpr,
):
    row = tl.program_id(0)
    cand = tl.arange(0, BLOCK)
    lane = cand < W * TOPK
    rank = cand // TOPK
    j = cand % TOPK
    # packs is [W, 2, rows, TOPK] int32 (one gathered pack per rank).
    score_off = ((rank * 2) * rows + row) * TOPK + j
    bits = tl.load(packs_ptr + score_off, mask=lane, other=-8388608)  # -inf bits
    # Order-preserving map of the fp32 bits onto [0, 2^32): integer order = float
    # order (negative floats flip all bits, others set the top bit). -inf maps to
    # 0x007FFFFF; padding lanes get 0 so they rank below every real candidate.
    ubits = bits.to(tl.int64) & 0xFFFFFFFF
    ordered = tl.where(bits < 0, 0xFFFFFFFF - ubits, ubits + 0x80000000)
    ordered = tl.where(lane, ordered, 0)

    # Radix select, 8 bits per pass: find the TOPK-th largest value `prefix` and
    # how many of the candidates equal to it are still needed.
    bins = tl.arange(0, 512)
    prefix = tl.zeros((), dtype=tl.int64)
    prefix_mask = tl.zeros((), dtype=tl.int64)
    need = tl.full((), TOPK, dtype=tl.int64)
    for p in tl.static_range(4):
        shift = 24 - 8 * p
        digit = (ordered >> shift) & 255
        match = (ordered & prefix_mask) == prefix
        # Non-matching candidates go to bins 256..511, which are ignored.
        hist = tl.histogram(tl.where(match, digit, 256 + digit).to(tl.int32), 512)
        hist = tl.where(bins < 256, hist, 0).to(tl.int64)
        total = tl.sum(hist)
        count_ge = total - tl.cumsum(hist, 0) + hist
        d = tl.max(tl.where((bins < 256) & (count_ge >= need), bins, -1))
        count_gt = tl.sum(tl.where(bins > d, hist, 0))
        need = need - count_gt
        prefix = prefix | (d.to(tl.int64) << shift)
        prefix_mask = prefix_mask | (255 << shift)

    above = ordered > prefix
    equal = ordered == prefix
    # Among ties at the threshold, the lowest candidate indices are kept.
    selected = above | (equal & (tl.cumsum(equal.to(tl.int32), 0) <= need))
    out_idx = tl.cumsum(selected.to(tl.int32), 0) - 1
    # 0x007FFFFF is the mapped -inf; a selected -inf candidate is padding.
    real = ordered > 0x007FFFFF
    pos_off = ((rank * 2 + 1) * rows + row) * TOPK + j
    pos = tl.load(packs_ptr + pos_off, mask=selected & real, other=-1).to(tl.int64)
    valid = selected & real & (pos >= 0)
    if HAS_PAGE_TABLE:
        safe = tl.where(valid, pos, 0)
        page_id = tl.load(
            page_table_ptr + row * page_table_stride + safe // PAGE,
            mask=valid,
            other=0,
        ).to(tl.int64)
        result = page_id * PAGE + safe % PAGE
    else:
        result = pos
    result = tl.where(valid, result, -1).to(tl.int32)
    tl.store(out_ptr + row * TOPK + out_idx, result, mask=selected)


def merge_candidates_triton(
    gathered: torch.Tensor,
    w: int,
    topk: int,
    *,
    page_table: Optional[torch.Tensor] = None,
    page: int = 1,
) -> torch.Tensor:
    """``merge_candidates`` and, given ``page_table``, ``positions_to_slots``.

    Returns [rows, topk] int32, -1 padded. The selected set equals the torch
    reference; the order is candidate order (rank-major), not score order.
    """
    rows = gathered.shape[1]
    out = torch.empty((rows, topk), dtype=torch.int32, device=gathered.device)
    if rows == 0:
        return out
    packs = gathered.contiguous()
    block = triton.next_power_of_2(w * topk)
    has_page_table = page_table is not None
    if has_page_table:
        assert page_table.stride(1) == 1
    _merge_candidates_kernel[(rows,)](
        packs,
        rows,
        page_table if has_page_table else packs,
        page_table.stride(0) if has_page_table else 0,
        out,
        TOPK=topk,
        W=w,
        BLOCK=block,
        PAGE=page,
        HAS_PAGE_TABLE=has_page_table,
    )
    return out


# ===========================================================================
# Public entry point
# ===========================================================================
def _aiter_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, seq_lens, block_tables, max_seq_len):
    """sglang kernels/ops/attention/dsa/paged_mqa_logits.py:95 aiter_paged_mqa_logits (preshuffle, block 64)."""
    from aiter.ops.triton.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits

    q = q_fp8.unsqueeze(1)
    batch_size, next_n, _, _ = q.shape
    logits = torch.empty((batch_size * next_n, max_seq_len), device=q.device, dtype=torch.float32)
    deepgemm_fp8_paged_mqa_logits(q, kv_cache_fp8, weights, logits, seq_lens, block_tables, max_seq_len,
                                  Preshuffle=True, KVBlockSize=PAGE_SIZE)
    return logits


def dcp_indexer_topk(
    q_fp8: torch.Tensor,            # [T, 32, 128] fp8_e4m3 index queries
    kv_cache_fp8: torch.Tensor,     # [num_pages, 64, 1, 132] uint8 view: preshuffled fp8 K + fp32 scale per page
    weights: torch.Tensor,          # [T, 32] fp32 per-head gate
    owned_tables: torch.Tensor,     # [T, ceil(P / 4)] int32: pages 0, 4, 8, ... of each row's page table
    owned_lens: torch.Tensor,       # [T] int32: tokens of [0, L) that fall in the owned pages
    real_page_table: torch.Tensor,  # [T, P] int32: the row's full page table (positions -> KV slots)
    other_packs: torch.Tensor,      # [3 * 2, T, 2048] int32: ranks 1..3's packs (stand-in for the all_gather)
    index_topk: int = 2048,
) -> torch.Tensor:
    """[T, 2048] int32 global KV slots of the global top-2048 (-1 padded); order = candidate order."""
    local_logits = _aiter_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, owned_lens, owned_tables,
                                           owned_tables.shape[1] * PAGE_SIZE)
    if local_logits.shape[1] < index_topk:
        local_logits = torch.nn.functional.pad(local_logits, (0, index_topk - local_logits.shape[1]),
                                               value=-float("inf"))
    from sgl_kernel import fast_topk_v2

    local_idx = fast_topk_v2(local_logits, owned_lens, index_topk)
    pack = pack_candidates_triton(local_logits, local_idx, owned_lens, index_topk, PAGE_SIZE, DCP, RANK)
    gathered = torch.cat([pack, other_packs], dim=0)          # rank order 0, 1, 2, 3
    return merge_candidates_triton(gathered, DCP, index_topk, page_table=real_page_table, page=PAGE_SIZE)
