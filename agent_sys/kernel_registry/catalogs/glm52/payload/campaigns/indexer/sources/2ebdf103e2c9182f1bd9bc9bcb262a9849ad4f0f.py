# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# (gluon kernel + host wrapper copied verbatim from aiter e7d2453; sglang glue copied from
#  sglang 11bfd868 -- see the provenance block below.)
"""GLM-5.2 DSA top-k indexer, TARGET_VERIFY decode leg, as run by sglang v0.5.21 on gfx950.

ONE public entry point: ``dsa_indexer_topk_paged`` = the computation inside
``Indexer._get_topk_paged`` (sglang python/sglang/srt/layers/attention/dsa/dsa_indexer.py:1191-1433)
for SGLANG_USE_AITER=1 + aiter preshuffle (triton >= 3.5) + dsa_topk_backend sgl-kernel +
SGLANG_DSA_FUSE_TOPK=1 (default) + SGLANG_OPT_USE_TOPK_V2=false (GLM-5.2 PD recipe):

  1. logits = aiter_paged_mqa_logits(q_fp8, kv_cache_fp8, weights, seqlens_expanded,
                                     real_page_table, max_seq_len, preshuffle=True, kv_block_size=64)
       -> aiter.ops.triton.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits(Preshuffle=True, KVBlockSize=64)
       -> gluon kernel _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle
  2. _mask_init_and_local_tokens: no-op for GLM-5.2 (index_init_tokens = index_local_tokens = 0)
  3. metadata.topk_transform -> DSATopKBackend.topk_transform -> sgl_kernel.fast_topk_transform_fused(
         score=logits, lengths=seqlens_expanded, page_table_size_1=page_table_1,
         cu_seqlens_q=arange(rows+1), topk=2048)

Provenance (verbatim copies, md5 of the image files):
  aiter/ops/triton/gluon/pa_mqa_logits.py      cf99858f1adb153c5ec09b753b6a3adb  lines 6-56, 411-1422
  aiter/ops/triton/attention/pa_mqa_logits.py  605138f571b0a985c235f744aedc77bd  _compile_deepgemm_fp8_paged_mqa_logits,
                                                deepgemm_fp8_paged_mqa_logits (reduced to the path taken here:
                                                gfx950, triton>=3.5 JIT gluon, Preshuffle=True, no VarCtx)
  sglang/kernels/ops/attention/dsa/paged_mqa_logits.py  d9feaecd...  aiter_paged_mqa_logits
"""
from __future__ import annotations

import os
from functools import cache

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.gluon.pa_decode_gluon import get_cdna_version

# Number of XCDs the phase-1 grid is round-robined over (gfx950 SPX = 8).
# Used to relabel program ids so that the KV-split pieces that share page-table
# entries land on the same XCD and hit its private L2 instead of the fabric.
# 1 == identity (control).
_NUM_XCD = tl.constexpr(int(os.environ.get("DSA_NUM_XCD", "8")))

try:
    from triton.experimental.gluon.language.amd.cdna3 import (
        s_set_prio as _amd_s_set_prio,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
    )
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_group_barrier as _amd_iglp_sched_group_barrier,
    )
except ImportError:
    # ignore iglp hint
    @gluon.jit
    def _amd_iglp_sched_barrier(inst_mask):
        pass

    @gluon.jit
    def _amd_iglp_sched_group_barrier(inst_mask, cnt, _):
        pass

    @gluon.jit
    def _amd_s_set_prio(prio):
        pass


# for some newer triton>=3.5 version, a 3D instr_shape is required.
try:
    _cdna_version = get_cdna_version()
    _version = _cdna_version if _cdna_version > 0 else 3
    _: gl.constexpr = gl.amd.AMDMFMALayout(
        version=_version,
        instr_shape=[16, 16],
        transposed=False,
        warps_per_cta=[1, 1],
        tiles_per_warp=[1, 1],
    )
    _Use_2d_instr_shape_mfma_layout = tl.constexpr(True)
except Exception:  # noqa: BLE001
    _Use_2d_instr_shape_mfma_layout = tl.constexpr(False)


@triton.jit
def _sum_combine(a, b):
    return a + b


@gluon.jit
def _dot_fp8(a, b, acc, CDNA_VERSION: gl.constexpr):
    # gfx950: the full-rate v_mfma_*_f8f6f4 instructions are only reachable through the
    # scaled-MFMA op (plain tt.dot has no fp8 mapping for K=64/128 shapes); unit e8m0
    # scales (0x7F == 2^0) make it a plain fp8 dot. The legacy v_mfma_f32_16x16x32_fp8
    # runs at half of gfx950's fp8 rate (16 cycles each, measured).
    # Epilogue relu: gl.maximum(.., propagate_nan=ALL) lowers to one v_maximum3_f32 per
    # element; the default maxnum form costs a canonicalize v_max plus the v_max.
    # Bitwise identical for finite inputs; a NaN MFMA output (only from fp8 NaN codes,
    # which the ue8m0 cache writer does not produce from finite K) now propagates
    # instead of being clipped to 0.
    if CDNA_VERSION == 4:
        out = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", acc)
    else:
        out = gl.amd.cdna3.mfma(a, b, acc)
    return out


# ===========================================================================
# Device side: aiter/ops/triton/gluon/pa_mqa_logits.py, preshuffle kernel (verbatim)
# ===========================================================================
@gluon.jit
def _gluon_deepgemm_fp8_paged_mqa_logits_preshuffle(
    batch_size,
    next_n,
    heads_num,
    Q_buffer,
    stride_q_batch,
    stride_q_next_n,
    stride_q_heads,
    KV_buffer,
    stride_k_seq,
    scale_buffer,
    stride_scale_seq,
    context_len_ptr,
    kv_indices,
    weights,
    stride_w_batch,
    OutLogits_buffer,
    stride_out_batch,
    max_model_len,
    max_block_len,
    num_block,
    SplitKV: gl.constexpr,
    ChunkQ: gl.constexpr,
    ChunkK: gl.constexpr,
    HiddenDim: gl.constexpr,
    KVBlockSize: gl.constexpr = 16,
    CDNA_VERSION: gl.constexpr = 3,
    ARCH: gl.constexpr = "gfx942",
):
    IS_GFX1250: gl.constexpr = ARCH == "gfx1250"
    # ===---------------------------------------------------
    # Gluon Layout
    # ===---------------------------------------------------
    if IS_GFX1250:
        NumWarps: gl.constexpr = 1
    else:
        NumWarps: gl.constexpr = 4
    ThreadsPerWarp: gl.constexpr = 32 if IS_GFX1250 else 64

    ValQMPerThread: gl.constexpr = ChunkQ // (
        NumWarps * ThreadsPerWarp // (HiddenDim // 16)
    )
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[ValQMPerThread, 16],  # q type is fp8 (E4M3)
        threads_per_warp=[ThreadsPerWarp // (HiddenDim // 16), HiddenDim // 16],
        warps_per_cta=[NumWarps, 1],
        order=[1, 0],
    )

    if IS_GFX1250:
        ChunkKPerStage: gl.constexpr = 16
    else:
        ChunkKPerStage: gl.constexpr = ChunkK // 2
    MFMAPerWarp: gl.constexpr = ChunkKPerStage // 16 // NumWarps

    if IS_GFX1250:
        FP8_K_DIM: gl.constexpr = 128 if HiddenDim > 64 else 64
        if NumWarps == 1:
            warp_bases: gl.constexpr = []
        elif NumWarps == 2:
            warp_bases: gl.constexpr = [[0, 1]]
        elif NumWarps == 4:
            warp_bases: gl.constexpr = [[0, 1], [0, 2]]
        else:
            warp_bases: gl.constexpr = [[0, 1], [0, 2], [0, 4]]
        mfma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
            version=3,
            transposed=False,
            instr_shape=[16, 16, FP8_K_DIM],
            warp_bases=warp_bases,
        )
    elif _Use_2d_instr_shape_mfma_layout:
        mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
            version=CDNA_VERSION,
            instr_shape=[16, 16],
            transposed=False,
            warps_per_cta=[1, NumWarps],
            tiles_per_warp=[1, MFMAPerWarp],
        )
    else:
        # gfx950, 32 heads: one 32x32x64 f8f6f4 tile per warp (32 heads x 32 tokens,
        # 2 MFMAs per stage). The head sum is then in-thread plus one permlane32 swap
        # (16x16 tiles need permlane32 + permlane16 rounds), and each lane stores one
        # logit. K addresses / weight / scale / store layouts all derive from
        # mfma_layout, so the preshuffled cache is read unchanged (4 x dwordx4 per lane
        # per stage). The head-sum order changes, so logits differ at rounding level;
        # driver SNR per case is unchanged from pristine. Other shapes: 16x16x128.
        if CDNA_VERSION == 4 and ChunkQ == 32 and ChunkKPerStage == 32 * NumWarps:
            mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
                version=CDNA_VERSION,
                instr_shape=[32, 32, 64],
                transposed=False,
                warps_per_cta=[1, NumWarps],
                tiles_per_warp=[1, 1],
            )
        else:
            mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
                version=CDNA_VERSION,
                instr_shape=[16, 16, 128] if CDNA_VERSION == 4 else [16, 16, 32],
                transposed=False,
                warps_per_cta=[1, NumWarps],
                tiles_per_warp=[1, MFMAPerWarp],
            )

    mfma_layout_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    mfma_layout_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    layout_scale: gl.constexpr = gl.SliceLayout(1, mfma_layout)

    if IS_GFX1250:
        NUM_BUFFERS: gl.constexpr = 2
        KV_SHARED: gl.constexpr = gl.SwizzledSharedLayout(
            vec=1, per_phase=1, max_phase=1, order=[1, 0]
        )
        kv_shared = gl.allocate_shared_memory(
            KV_buffer.type.element_ty,
            [NUM_BUFFERS, 1, KVBlockSize * HiddenDim],
            KV_SHARED,
        )
        kv_scale_shared = gl.allocate_shared_memory(
            scale_buffer.type.element_ty,
            [NUM_BUFFERS, 1, KVBlockSize * HiddenDim // 128],
            KV_SHARED,
        )

        pid = tl.program_id(0)
        pid_batch, remain_pid = pid % batch_size, pid // batch_size
        pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n
        context_length = gl.load(context_len_ptr + pid_batch)

        context_chunk_num = tl.cdiv(context_length, ChunkK)
        split_context_chunk_num = context_chunk_num // SplitKV
        residual_context_chunks = context_chunk_num % SplitKV
        split_context_start = (
            pid_split_kv * split_context_chunk_num * ChunkK
            + min(pid_split_kv, residual_context_chunks) * ChunkK
        )
        split_context_length = min(
            context_length - split_context_start,
            split_context_chunk_num * ChunkK
            + (ChunkK if pid_split_kv < residual_context_chunks else 0),
        )
        if split_context_length <= 0:
            return

        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )
        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        mfma_q = gl.convert_layout(q, mfma_layout_a)
        zero_acc = gl.zeros((ChunkQ, KVBlockSize), dtype=tl.float32, layout=mfma_layout)
        col = gl.arange(0, KVBlockSize, layout=gl.SliceLayout(0, mfma_layout))

        n_blocks = tl.cdiv(split_context_length, KVBlockSize)
        cblk_base = split_context_start // KVBlockSize
        kv_idx_base = kv_indices + pid_batch * max_block_len + cblk_base

        # Prologue: kick off the async_load for block 0 into buffer 0.
        save_blk_idx: gl.int32 = n_blocks - 1
        blk_cur = gl.load(kv_idx_base)
        blk_next = gl.load(kv_idx_base + gl.minimum(1, save_blk_idx))
        desc_cur = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=KV_buffer,
            shape=(num_block, KVBlockSize * HiddenDim),
            strides=(stride_k_seq, 1),
            block_shape=(1, KVBlockSize * HiddenDim),
            layout=KV_SHARED,
        )
        BLOCKSCALE_SIZE: gl.constexpr = 128
        assert HiddenDim // BLOCKSCALE_SIZE == 1
        desc_scale_cur = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=scale_buffer,
            shape=(num_block, KVBlockSize * HiddenDim // BLOCKSCALE_SIZE),
            strides=(stride_scale_seq, 1),
            block_shape=(1, KVBlockSize * HiddenDim // BLOCKSCALE_SIZE),
            layout=KV_SHARED,
        )
        gl.amd.gfx1250.tdm.async_load(desc_cur, [blk_cur, 0], kv_shared.index(0))
        gl.amd.gfx1250.tdm.async_load(
            desc_scale_cur, [blk_cur, 0], kv_scale_shared.index(0)
        )

        for j in range(n_blocks):
            buf = j % NUM_BUFFERS
            context_idx = split_context_start + j * KVBlockSize
            # blk = blk_cur
            # k_scale_f = gl.amd.cdna3.buffer_load(
            #     ptr=scale_buffer,
            #     offsets=blk * stride_scale_seq + col,
            # )

            # Prefetch block j+1 into the other buffer before waiting on block j.
            if j + 1 < n_blocks:
                blk_cur = blk_next
                blk_next = gl.load(kv_idx_base + gl.minimum(j + 2, save_blk_idx))
                gl.amd.gfx1250.tdm.async_load(
                    desc_cur, [blk_cur, 0], kv_shared.index((j + 1) % NUM_BUFFERS)
                )
                gl.amd.gfx1250.tdm.async_load(
                    desc_scale_cur,
                    [blk_cur, 0],
                    kv_scale_shared.index((j + 1) % NUM_BUFFERS),
                )
                # Leave exactly the just-issued load (block j+1) in flight.
                gl.amd.gfx1250.tdm.async_wait(3)
            else:
                gl.amd.gfx1250.tdm.async_wait(1)

            K_WIDTH: gl.constexpr = 16
            mfma_k = (
                kv_shared.index(buf)
                .reshape(
                    (
                        KVBlockSize // 16,
                        HiddenDim // (2 * K_WIDTH),
                        2,
                        16,
                        K_WIDTH,
                    )
                )
                .permute((0, 3, 1, 2, 4))
                .reshape((KVBlockSize, HiddenDim))
                .permute([1, 0])
                .load(layout=mfma_layout_b)
            )
            o = gl.amd.gfx1250.wmma(mfma_q, mfma_k, zero_acc)

            if j + 1 < n_blocks:
                gl.amd.gfx1250.tdm.async_wait(2)
            else:
                gl.amd.gfx1250.tdm.async_wait(0)

            k_scale_f = (
                kv_scale_shared.index(buf)
                .reshape((1, KVBlockSize))
                .load(layout=mfma_layout)
            )
            o = o * k_scale_f
            o = gl.maximum(o, 0.0)
            o = o * scale_weight[:, None]

            valid = (context_idx + col) <= (context_length - next_n + pid_next_n)
            o = tl.where(valid[None, :], o, float("-inf"))
            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine)  # gfx1250 path

            store_off = context_idx + col
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=store_off,
                mask=(store_off >= 0) & (store_off < max_model_len),
            )
        return

    ContextBlockPerChunkK: gl.constexpr = ChunkK // KVBlockSize
    ChunkKStagePerContextBlock: gl.constexpr = KVBlockSize // ChunkKPerStage

    LoadBlockIndiceForEachStage: gl.constexpr = ChunkKPerStage % KVBlockSize == 0
    # Rows per shuffled group, matching the `layout=(ShuffleRows, 16)` the host
    # passed to shuffle_weight. A page shorter than the 16-token MFMA tile can
    # only be shuffled in groups of its own length, so the tile is then read
    # from 16 // KVBlockSize pages -- which the per-lane page index already does.
    ShuffleRows: gl.constexpr = min(KVBlockSize, 16)

    # DS_WRITE: gl.constexpr = 0x200
    DS_READ: gl.constexpr = 0x100
    BUFFER_LOAD: gl.constexpr = 0x020
    MFMA: gl.constexpr = 0x008
    # VALU: gl.constexpr = 0x002

    # ===---------------------------------------------------
    # Mapping WorkTile
    # ===---------------------------------------------------
    pid = tl.program_id(0)

    # XCD-local swizzle: the hardware round-robins workgroup i onto XCD i % NUM_XCD,
    # so the rows that share a page table (the 6 TARGET_VERIFY siblings of a request)
    # get scattered over 6 private L2s. Relabelling the program id so that each XCD
    # owns a contiguous slab of (split_kv-major, row-minor) work puts every row of a
    # given KV split on one XCD. This is a bijection over the whole grid, so the
    # logits are bitwise identical.
    NUM_XCD: gl.constexpr = _NUM_XCD
    if NUM_XCD > 1:
        grid_size = batch_size * next_n * SplitKV
        wg_per_xcd = grid_size // NUM_XCD
        if wg_per_xcd * NUM_XCD == grid_size:
            pid = (pid % NUM_XCD) * wg_per_xcd + pid // NUM_XCD

    # ===---------------------------------------------------
    pid_batch, remain_pid = pid % batch_size, pid // batch_size
    pid_next_n, pid_split_kv = remain_pid % next_n, remain_pid // next_n
    # ===---------------------------------------------------
    context_length = gl.load(context_len_ptr + pid_batch)

    context_chunk_num = tl.cdiv(context_length, ChunkK)
    split_context_chunk_num = context_chunk_num // SplitKV
    residual_context_chunks = context_chunk_num % SplitKV
    split_context_start = (
        pid_split_kv * split_context_chunk_num * ChunkK
        + min(pid_split_kv, residual_context_chunks) * ChunkK
    )
    split_context_length = min(
        context_length - split_context_start,
        split_context_chunk_num * ChunkK
        + (ChunkK if pid_split_kv < residual_context_chunks else 0),
    )

    if split_context_length <= 0:
        return

    if LoadBlockIndiceForEachStage:
        split_context_block = tl.cdiv(split_context_length, KVBlockSize)
        split_context_length = split_context_block * KVBlockSize

        residual_context_blocks = (
            ContextBlockPerChunkK - split_context_block % ContextBlockPerChunkK
        ) % ContextBlockPerChunkK
        residual_context = residual_context_blocks * KVBlockSize

        # ===---------------------------------------------------
        # Pipeline Start
        _amd_iglp_sched_barrier(0x0)
        # ===---------------------------------------------------
        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_idx = split_context_start - residual_context

        mask_kv_next_0 = (
            context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + context_idx // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_0,
        )

        mask_kv_next_1 = (
            (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize
        ) >= split_context_start // KVBlockSize
        context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
            ptr=kv_indices,
            offsets=pid_batch * max_block_len
            + (context_idx + ChunkKPerStage) // KVBlockSize
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            // KVBlockSize,
            mask=mask_kv_next_1,
        )

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * (ShuffleRows * 16)
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % ShuffleRows
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // ShuffleRows
            * ShuffleRows
            * HiddenDim
        )[
            None, :
        ]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        context_kv_idx_next_0 = tl.where(mask_kv_next_0, context_kv_idx_next_0, 0)
        k_next_0 = gl.load(
            KV_buffer
            + offset_k_fixed
            + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
        )
        k_scale_f_next_0 = gl.load(
            scale_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        if context_idx + ChunkK < split_context_start + split_context_length:
            context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        context_kv_idx_next_1 = tl.where(mask_kv_next_1, context_kv_idx_next_1, 0)
        k_next_1 = gl.load(
            KV_buffer
            + offset_k_fixed
            + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
        )
        k_scale_f_next_1 = gl.load(
            scale_buffer
            + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize,
        )

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
        _amd_s_set_prio(1)

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        store_cols = context_idx + gl.arange(
            0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
        )
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=store_cols,
            mask=(store_cols >= split_context_start) & (store_cols < max_model_len),
        )

        for context_idx in range(
            split_context_start - residual_context,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            context_kv_idx_next_1 = gl.amd.cdna3.buffer_load(
                ptr=kv_indices,
                offsets=pid_batch * max_block_len
                + (context_idx + ChunkK + ChunkKPerStage) // KVBlockSize
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                // KVBlockSize,
            )
            k_next_0 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_0.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_0 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            store_cols = (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            )
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=store_cols,
                mask=(store_cols >= split_context_start) & (store_cols < max_model_len),
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            # #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            # #!=----------------------------
            if (
                context_idx + ChunkK + ChunkK
                < split_context_start + split_context_length
            ):
                context_kv_idx_next_0 = gl.amd.cdna3.buffer_load(
                    ptr=kv_indices,
                    offsets=pid_batch * max_block_len
                    + (context_idx + ChunkK + ChunkK) // KVBlockSize
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                    // KVBlockSize,
                )
            k_next_1 = gl.load(
                KV_buffer
                + offset_k_fixed
                + context_kv_idx_next_1.to(gl.int64)[None, :] * stride_k_seq
            )
            k_scale_f_next_1 = gl.load(
                scale_buffer
                + context_kv_idx_next_1.to(gl.int64) * stride_scale_seq
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
                % KVBlockSize,
            )
            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
            _amd_s_set_prio(0)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=(
                    context_idx
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=(
                    context_idx
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                )
                < max_model_len,
            )

        context_idx = split_context_start + split_context_length - ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        logits = tl.where(mask, logits, float("-inf"))
        store_cols = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
        )
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=store_cols,
            mask=(store_cols >= split_context_start) & (store_cols < max_model_len),
        )
    else:
        context_idx = split_context_start
        current_chunk_rank = context_idx // ChunkKPerStage % ChunkKStagePerContextBlock
        block_idx = context_idx // KVBlockSize
        batch_blocks = tl.cdiv(context_length, KVBlockSize)

        q = gl.amd.cdna3.buffer_load(
            ptr=Q_buffer,
            offsets=pid_batch * stride_q_batch
            + pid_next_n * stride_q_next_n
            + (
                gl.arange(0, ChunkQ, layout=gl.SliceLayout(1, layout_q))
                * stride_q_heads
            )[:, None]
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(0, layout_q))[None, :],
        )

        context_kv_idx_next_0 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx
        )
        block_idx += 1

        context_kv_idx_next_1 = gl.load(
            kv_indices + pid_batch * max_block_len + block_idx,
            mask=block_idx < batch_blocks,
        )
        block_idx += 1

        scale_weight = gl.amd.cdna3.buffer_load(
            ptr=weights,
            offsets=(pid_batch * next_n + pid_next_n) * stride_w_batch
            + gl.arange(0, ChunkQ, layout=layout_scale),
        )

        offset_k_fixed = (
            gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b)) % 16
            + gl.arange(0, HiddenDim, layout=gl.SliceLayout(1, mfma_layout_b))
            // 16
            * (ShuffleRows * 16)
        )[:, None] + (
            gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % ShuffleRows
            * 16
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            % KVBlockSize
            // ShuffleRows
            * ShuffleRows
            * HiddenDim
        )[
            None, :
        ]

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        mfma_q = gl.convert_layout(q, mfma_layout_a)

        k_next_0 = gl.load(
            KV_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
            + offset_k_fixed
            + context_idx % KVBlockSize * HiddenDim
        )
        k_scale_f_next_0 = gl.load(
            scale_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
            + (
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )

        _amd_iglp_sched_group_barrier(DS_READ, 4, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 4, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(DS_READ, 2, 0)

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        # ===---------------------------------------------------
        # Precompute First Iteration
        # ===---------------------------------------------------
        zero = gl.zeros((ChunkQ, ChunkKPerStage), dtype=tl.float32, layout=mfma_layout)

        k = k_next_0
        k_scale_f = k_scale_f_next_0

        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------
        k_next_1 = gl.load(
            KV_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
            + offset_k_fixed
            + (context_idx + ChunkKPerStage) % KVBlockSize * HiddenDim
        )
        k_scale_f_next_1 = gl.load(
            scale_buffer
            + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
            + (
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b))
            )
            % KVBlockSize,
        )
        current_chunk_rank += 2

        _amd_s_set_prio(3)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        _amd_iglp_sched_group_barrier(MFMA, 8, 0)
        _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
        #!=----------------------------
        _amd_iglp_sched_barrier(0x0)
        #!=----------------------------

        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

        o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
        _amd_s_set_prio(1)

        mask = (
            context_idx
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=(
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=(
                context_idx
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            )
            < max_model_len,
        )

        for context_idx_ in range(
            split_context_start,
            split_context_start + split_context_length - ChunkK,
            ChunkK,
        ):
            k = k_next_1
            k_scale_f = k_scale_f_next_1

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            if current_chunk_rank == ChunkKStagePerContextBlock:
                current_chunk_rank = 0
                context_kv_idx_next_0 = context_kv_idx_next_1
                context_kv_idx_next_1 = gl.load(
                    kv_indices + pid_batch * max_block_len + block_idx,
                    mask=block_idx < batch_blocks,
                )
                block_idx += 1

            k_next_0 = gl.load(
                KV_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
                + offset_k_fixed
                + (context_idx_ + ChunkK) % KVBlockSize * HiddenDim
            )
            k_scale_f_next_0 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize,
            )

            _amd_s_set_prio(3)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
            o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
            _amd_s_set_prio(1)

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=(
                    context_idx_
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=(
                    context_idx_
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                )
                < max_model_len,
            )

            # =======================================================================================

            k = k_next_0
            k_scale_f = k_scale_f_next_0

            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------
            k_next_1 = gl.load(
                KV_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_k_seq
                + offset_k_fixed
                + (context_idx_ + ChunkK + ChunkKPerStage) % KVBlockSize * HiddenDim
            )
            k_scale_f_next_1 = gl.load(
                scale_buffer
                + context_kv_idx_next_0.to(gl.int64) * stride_scale_seq
                + (
                    context_idx_
                    + ChunkK
                    + ChunkKPerStage
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout_b)
                    )
                )
                % KVBlockSize
            )
            current_chunk_rank += 2

            _amd_s_set_prio(2)
            mfma_k = gl.convert_layout(k, mfma_layout_b)
            o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)

            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            _amd_iglp_sched_group_barrier(BUFFER_LOAD, 2, 0)
            _amd_iglp_sched_group_barrier(MFMA, 8, 0)
            #!=----------------------------
            _amd_iglp_sched_barrier(0x0)
            #!=----------------------------

            k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))

            o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
            _amd_s_set_prio(0)

            mask = (
                context_idx_
                + ChunkK
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
                <= context_length - next_n + pid_next_n
            )

            logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
            logits = tl.where(mask, logits, float("-inf"))
            gl.amd.cdna3.buffer_store(
                logits,
                ptr=OutLogits_buffer
                + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
                offsets=(
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                ),
                mask=(
                    context_idx_
                    + ChunkK
                    + gl.arange(
                        0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout)
                    )
                )
                < max_model_len,
            )
            context_idx = context_idx_ + ChunkK

        k = k_next_1
        k_scale_f = k_scale_f_next_1

        _amd_s_set_prio(1)
        mfma_k = gl.convert_layout(k, mfma_layout_b)
        o = _dot_fp8(mfma_q, mfma_k, zero, CDNA_VERSION)
        k_scale_f = gl.convert_layout(k_scale_f, gl.SliceLayout(0, mfma_layout))
        o = gl.maximum(o, 0.0, propagate_nan=tl.PropagateNan.ALL) * scale_weight[:, None]
        _amd_s_set_prio(0)

        mask = (
            context_idx
            + ChunkKPerStage
            + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            <= context_length - next_n + pid_next_n
        )

        logits = gl.reduce(o, axis=0, combine_fn=_sum_combine) * k_scale_f
        logits = tl.where(mask, logits, float("-inf"))
        gl.amd.cdna3.buffer_store(
            logits,
            ptr=OutLogits_buffer
            + (pid_batch * next_n + pid_next_n).to(tl.int64) * stride_out_batch,
            offsets=(
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            ),
            mask=(
                context_idx
                + ChunkKPerStage
                + gl.arange(0, ChunkKPerStage, layout=gl.SliceLayout(0, mfma_layout))
            )
            < max_model_len,
        )



# ===========================================================================
# Host side (aiter/ops/triton/attention/pa_mqa_logits.py, reduced to this path)
# ===========================================================================
from triton.backends.compiler import GPUTarget  # noqa: E402
from triton.experimental.gluon._runtime import GluonASTSource as ASTSource  # noqa: E402

from aiter import dtypes  # noqa: E402
from aiter.jit.utils.chip_info import get_gfx  # noqa: E402
from aiter.ops.triton.utils.device_info import get_num_sms  # noqa: E402


@cache
def _compile_deepgemm_fp8_paged_mqa_logits(
    ChunkQ,
    ChunkK,
    Preshuffle,
    KVBlockSize,
    HiddenDim,
    is_padded_mode: bool,
    WavePerEU: int = 2,
    WavesPerEuHint: int = 2,
):
    gfx_version = get_gfx()
    assert gfx_version == "gfx950", gfx_version
    assert Preshuffle
    cdna_version = get_cdna_version()
    warp_size = 64
    target = GPUTarget("hip", gfx_version, warp_size)
    gfx_fp8_pointer = "*fp8e4nv"

    fn_signature = {
        "batch_size": "i32",
        "next_n": "i32",
        "heads_num": "i32",
        "Q_buffer": gfx_fp8_pointer,
        "stride_q_batch": "i32",
        "stride_q_next_n": "i32",
        "stride_q_heads": "i32",
        "KV_buffer": gfx_fp8_pointer,
        "stride_k_seq": "i32",
        "scale_buffer": "*fp32",
        "stride_scale_seq": "i32",
        "context_len_ptr": "*i32",
        "kv_indices": "*i32",
        "weights": "*fp32",
        "stride_w_batch": "i32",
        "OutLogits_buffer": "*fp32",
        "stride_out_batch": "i64",
        "max_model_len": "i32",
        "max_block_len": "i32",
        "num_block": "i32",
        "SplitKV": "i32",
    }
    fn_signature["ChunkQ"] = "constexpr"
    fn_signature["ChunkK"] = "constexpr"
    fn_signature["KVBlockSize"] = "constexpr"
    fn_signature["HiddenDim"] = "constexpr"
    fn_signature["CDNA_VERSION"] = "constexpr"
    fn_signature["ARCH"] = "constexpr"

    options = {
        "num_warps": 4,
        "waves_per_eu": WavesPerEuHint,
        "num_stages": 2,
        "num_ctas": 1,
        "cluster_dims": [1, 1, 1],
        "arch": gfx_version,
        "backend_name": "hip",
        "warp_size": warp_size,
        "name": "_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle",
    }

    kv_cache_attr = []
    if is_padded_mode:
        kv_cache_attr.append(["tt.divisibility", 16])

    src = ASTSource(
        fn=_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle,
        signature=fn_signature,
        constexprs={
            "ChunkQ": ChunkQ,
            "ChunkK": ChunkK,
            "KVBlockSize": KVBlockSize,
            "HiddenDim": HiddenDim,
            "CDNA_VERSION": cdna_version,
            "ARCH": gfx_version,
        },
        attrs={
            (2,): [["tt.divisibility", 16]],  # heads_num
            (3,): [["tt.divisibility", 16], ["tt.pointer_range", 32]],  # Q_buffer
            (4,): [["tt.divisibility", 16]],  # stride_q_batch
            (5,): [["tt.divisibility", 16]],  # stride_q_next_n
            (6,): [["tt.divisibility", 16]],  # stride_q_heads
            (7,): kv_cache_attr,  # KV_buffer
            (8,): kv_cache_attr,  # stride_k_seq
            (9,): kv_cache_attr,  # scale_buffer
            (10,): kv_cache_attr,  # stride_scale_seq
            (11,): [["tt.pointer_range", 32]],  # context_len_ptr
            (12,): [["tt.pointer_range", 32]],  # kv_indices
            (13,): [
                ["tt.divisibility", 16],
                ["tt.pointer_range", 32],
            ],  # weights
            (14,): [["tt.divisibility", 16]],  # stride_w_batch
            # OutLogits_buffer: no tt.pointer_range 32 (row offsets of wide logits exceed 2**31 B)
        },
    )
    return triton.compile(src, target=target, options=options)


# KV-split granularity multiplier (see SplitKV below). Re-swept on this driver once
# the XCD-local program-id swizzle was in place: the optimum moves down from 32 to
# 4-8 (median_ms 1:0.368, 2:0.322, 4:0.3033, 6:0.3030, 8:0.3044, 12:0.3103,
# 16:0.3208, 24:0.3341, 32:0.3527, 48:0.3894) because the swizzle already fixes the
# makespan imbalance that the fine split was papering over, and coarser pieces
# amortize the per-piece prologue. SplitKV = 80 at mul=8 keeps grid % 8 == 0 for
# any row count, which the swizzle needs.
_SPLIT_MUL = int(os.environ.get("DSA_SPLIT_MUL", "8"))

# Occupancy hint passed to the Triton AMD backend, decoupled from `WavePerEU` (which
# also feeds the SplitKV partition arithmetic -- changing that symbol would silently
# move SplitKV off 80 and break the `rows*SplitKV % 8 == 0` XCD-swizzle guard).
# The compiled kernel uses 92 VGPRs but the backend emits amdgpu-waves-per-eu="2,2"
# and reserves 169, capping occupancy at 2 waves/SIMD with zero LDS in play.
# Swept on this driver (per-case ms, bs8/10/12/14): 2 -> .2482/.2467/.2941/.2956,
# 3 -> .2304/.2289/.2673/.2743, 4 -> .2268/.2230/.2622/.2657, 5 -> .2305/.2227/
# .2628/.2652. 4 is the knee; 6 would budget 512/6 = 85 < 92 VGPR and spill.
_WAVES_PER_EU = int(os.environ.get("DSA_WAVES_PER_EU", "4"))


def deepgemm_fp8_paged_mqa_logits(
    q_fp8: torch.Tensor,  # [batch, next_n, heads, hidden] fp8 e4m3fn
    kv_cache,  # [num_pages, KVBlockSize, 1, hidden + 4] (preshuffled fp8 + fp32 scales per page)
    weights: torch.Tensor,  # [batch * next_n, heads] fp32
    out_logits: torch.Tensor,  # [batch * next_n, >= max_model_len] fp32
    context_lens: torch.Tensor,
    kv_indices: torch.Tensor,
    max_model_len: int,
    Preshuffle: bool = False,
    KVBlockSize: int = 1,
    ChunkK: int = 256,
    TotalCuCount: int | None = None,
    WavePerEU: int = 2,
):
    if TotalCuCount is None:
        TotalCuCount = get_num_sms()
    batch_size, next_n, heads, hidden_dim = q_fp8.size()
    _, block_Size, _, index_dim = kv_cache.size()
    _, max_block_len = kv_indices.size()

    TileQCount = batch_size * next_n
    # Decouple KV split granularity from the waves_per_eu launch hint: the stock
    # formula yields SplitKV = 10 for every batch size here, so every program starts
    # at t=0 and the makespan equals the longest row (max/mean imbalance 1.8-2.4x).
    # A finer split gives the scheduler smaller pieces to backfill with.
    SplitKV = (max(1, TotalCuCount // TileQCount) + 4) // 5 * 5 * WavePerEU * _SPLIT_MUL

    assert ChunkK % KVBlockSize == 0 or KVBlockSize % ChunkK == 0
    assert block_Size == KVBlockSize
    assert Preshuffle and KVBlockSize % 16 == 0

    kv_cache = kv_cache.view(-1, KVBlockSize * index_dim)
    num_block = kv_cache.shape[0]
    kv_cache_fp8, kv_cache_scale = (
        kv_cache[..., : KVBlockSize * hidden_dim],
        kv_cache[..., KVBlockSize * hidden_dim :],
    )
    kv_cache_fp8 = kv_cache_fp8.view(dtypes.fp8)
    kv_cache_scale = kv_cache_scale.view(torch.float32)

    grid = (batch_size * next_n * SplitKV, 1, 1)
    is_padded_mode = kv_cache_fp8.stride(0) % 16 == 0
    kernel = _compile_deepgemm_fp8_paged_mqa_logits(
        ChunkQ=heads,
        ChunkK=ChunkK,
        Preshuffle=Preshuffle,
        KVBlockSize=KVBlockSize,
        HiddenDim=hidden_dim,
        is_padded_mode=is_padded_mode,
        WavePerEU=WavePerEU,
        WavesPerEuHint=_WAVES_PER_EU,
    )
    cdna_version = get_cdna_version()
    kernel[grid](
        batch_size,
        next_n,
        heads,
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        q_fp8.stride(2),
        kv_cache_fp8,
        kv_cache_fp8.stride(0),
        kv_cache_scale,
        kv_cache_scale.stride(0),
        context_lens,
        kv_indices,
        weights,
        weights.stride(0),
        out_logits,
        out_logits.stride(0),
        max_model_len,
        max_block_len,
        num_block,
        SplitKV,
        # constexpr
        heads,
        ChunkK,
        KVBlockSize,
        hidden_dim,
        cdna_version,
        get_gfx(),
    )


# ===========================================================================
# sglang glue (kernels/ops/attention/dsa/paged_mqa_logits.py: aiter_paged_mqa_logits)
# ===========================================================================
def aiter_paged_mqa_logits(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seq_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_seq_len: int,
    *,
    preshuffle: bool,
    kv_block_size: int,
) -> torch.Tensor:
    q_fp8 = q_fp8.unsqueeze(1)
    batch_size, next_n, _, _ = q_fp8.shape
    logits = torch.empty(
        (batch_size * next_n, max_seq_len),
        device=q_fp8.device,
        dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        logits,
        seq_lens,
        block_tables,
        max_seq_len,
        Preshuffle=preshuffle,
        KVBlockSize=kv_block_size,
    )
    return logits


# Row-chunk size for the compiled top-k op (see topk_transform_fused). 0 disables.
_TOPK_ROW_CHUNK = int(os.environ.get("DSA_TOPK_ROW_CHUNK", "60"))

# ===========================================================================
# In-file single-pass top-k selector (replaces the compiled sgl_kernel
# coop_topk path).  Three small kernels (est / select Triton, collect Gluon):
#   K1 est     : per row, sample ~8192 logits (contiguous 128-element probes)
#                and binary-search (16 bits, rounding down) the sample key that
#                leaves ~OVER*topk elements above it in the full row -> a per-row
#                threshold; program 0 also builds the chunk prefix
#   K2 collect : ONE full pass over the logits; every key >= threshold is
#                appended to a per-row candidate buffer (volume-partitioned
#                over a fixed persistent grid of row-aligned chunks)
#   K3 select  : per row, hold the (few thousand) candidates in registers and
#                binary-search the exact k-th largest key, then emit exactly
#                min(L, topk) slots through the page table
# tl.histogram / tl.sort are sort-based on gfx950 (~300 us per 13 M elements),
# so the selector uses only loads, compares, sums and scans.
# The launch shape depends only on `rows` (CUDA-graph safe).
# ===========================================================================
_TOPK_T = int(os.environ.get("DSA_TOPK_T", "8192"))        # elements per chunk
_TOPK_GRID = int(os.environ.get("DSA_TOPK_GRID", "2048"))  # persistent programs
_TOPK_CAND = int(os.environ.get("DSA_TOPK_CAND", "8192"))  # candidates/row cap
_TOPK_OVER = int(os.environ.get("DSA_TOPK_OVER", "2"))      # candidate safety x topk
_TOPK_IMPL = os.environ.get("DSA_TOPK_IMPL", "triton")
_SMP_CH = 128                                               # probe length
_SMP_NCH = 64                                               # probes per row
_RMAX = 128
_CSTR = int(os.environ.get("DSA_TOPK_CSTR", "64"))  # int32 gap between rows' counters
_COLLECT_NW = int(os.environ.get("DSA_TOPK_COLLECT_NW", "4"))  # collect warps


@triton.jit
def _okey(b):
    # order-preserving int32 key of a float32 bit pattern
    return tl.where(b >= 0, b, (~b) ^ (-2147483648))


@triton.jit
def _topk_est(logits_ptr, lengths_ptr, pref_ptr, ctr_ptr, stride_l, rows,
              CH: tl.constexpr, NCH: tl.constexpr, TOPK: tl.constexpr,
              OVER: tl.constexpr, T: tl.constexpr, RMAX: tl.constexpr,
              CSTR: tl.constexpr):
    r = tl.program_id(0)
    # program 0 also builds the chunk prefix (was a separate 5 us dispatch)
    if r == 0:
        ro = tl.arange(0, RMAX)
        msk = ro < rows
        La = tl.load(lengths_ptr + ro, mask=msk, other=0)
        cp = tl.cumsum((La + (T - 1)) // T, 0)
        tl.store(pref_ptr + 1 + ro, cp, mask=msk)
        tl.store(pref_ptr, 0)
    tl.store(ctr_ptr + r * CSTR + 1, 0)
    L = tl.load(lengths_ptr + r)
    k = tl.minimum(L, TOPK)
    span = tl.maximum(CH, L // NCH)
    o = tl.arange(0, CH * NCH)
    p = (o // CH) * span + (o % CH)
    m = p < L
    b = tl.load(logits_ptr + r.to(tl.int64) * stride_l + p, mask=m, other=-1)
    key = _okey(b)
    ns = tl.sum(m.to(tl.int32), 0)
    t = tl.maximum(1, tl.minimum((OVER * k * ns) // L, ns))
    thr = 0
    if tl.sum((key >= 0).to(tl.int32), 0) < t:
        thr = -2147483648
    # bits 30..15 only: the threshold rounds down by < 2**15 key units (< 0.4 % of the
    # value), which can only add candidates -- every true top-k key stays above it.
    for i in tl.static_range(0, 16):
        cand = thr + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= t:
            thr = cand
    tl.store(ctr_ptr + r * CSTR, thr)


@gluon.jit
def _topk_collect(logits_ptr, lengths_ptr, pref_ptr, ctr_ptr, ckey_ptr,
                  cidx_ptr, rows, stride_l, T: gl.constexpr,
                  CAND: gl.constexpr, RMAX: gl.constexpr, GRID: gl.constexpr,
                  CSTR: gl.constexpr, NW: gl.constexpr):
    # Thread-local compaction with a pinned layout: thread t of the CTA owns the R
    # 16-byte vectors (a, t, 0..3) of the chunk. The slot of a selected element is
    # base + (count of the threads before t) + (its rank inside thread t): per-thread
    # sums and in-thread ranks are register ops, the NT-wide scan (one value per
    # thread) is the only cross-thread step, and the scattered stores keep the load
    # layout (no LDS round trip). Within-chunk order is a permutation -- free, since
    # the selector's output order is unconstrained and cross-chunk order was already
    # atomic arrival order.
    NT: gl.constexpr = 64 * NW
    R: gl.constexpr = T // (NT * 4)
    L3: gl.constexpr = gl.BlockedLayout([R, 1, 4], [1, 64, 1], [1, NW, 1], [2, 1, 0])
    LRT: gl.constexpr = gl.SliceLayout(2, L3)
    L2: gl.constexpr = gl.BlockedLayout([R, 1], [1, 64], [1, NW], [1, 0])
    L1: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    pid = gl.program_id(0)
    ro = gl.arange(0, RMAX, layout=L1)
    pref = gl.load(pref_ptr + 1 + ro, mask=ro < rows, other=2147483647)
    total = gl.load(pref_ptr + rows)
    a = gl.arange(0, R, layout=gl.SliceLayout(1, LRT))
    t = gl.arange(0, NT, layout=gl.SliceLayout(0, LRT))
    v = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    va = gl.expand_dims(gl.expand_dims(a, 1) * (NT * 4) + gl.expand_dims(t, 0) * 4, 2)
    loc = va + gl.expand_dims(gl.expand_dims(v, 0), 1)
    for c in range(pid, total, GRID):
        r = gl.sum((c >= pref).to(gl.int32), 0)
        start = (c - gl.load(pref_ptr + r)) * T
        L = gl.load(lengths_ptr + r)
        thr = gl.load(ctr_ptr + r * CSTR)
        idx = start + loc
        # 16-byte-granular load mask (rows are 4-aligned and >= L wide, so the tail
        # vector's extra elements are in-bounds); the exact `idx < L` gates selection.
        # Buffer base = the kernel argument (uniform), row folded into the 32-bit offset
        # (host asserts the logits fit in 2**31 bytes): a base derived from the reduced
        # `r` is not provably uniform and every buffer op becomes a waterfall loop.
        b = gl.amd.cdna3.buffer_load(ptr=logits_ptr, offsets=r * stride_l + idx,
                                     mask=(start + va) < L, other=-1)
        key = gl.where(b >= 0, b, (~b) ^ (-2147483648))
        si = ((idx < L) & (key >= thr)).to(gl.int32)
        s_v = gl.sum(si, 2)
        cnt = gl.sum(s_v, 0)
        n = gl.sum(cnt, 0)
        if n > 0:
            base = gl.atomic_add(ctr_ptr + r * CSTR + 1, n, sem="relaxed")
            # tt.scan only lowers on blocked layouts: L1 / L2 are the same thread maps
            # as the slice layouts (no data movement).
            cnt_b = gl.convert_layout(cnt, L1)
            tb = gl.convert_layout(gl.associative_scan(cnt_b, 0, _sum_combine) - cnt_b,
                                   gl.SliceLayout(0, LRT))
            s_vb = gl.convert_layout(s_v, L2)
            rv = gl.convert_layout(gl.associative_scan(s_vb, 0, _sum_combine) - s_vb, LRT)
            rk = gl.expand_dims(rv, 2) + (gl.associative_scan(si, 2, _sum_combine) - si)
            p = base + gl.expand_dims(gl.expand_dims(tb, 0), 2) + rk
            mm = (si != 0) & (p < CAND)
            gl.amd.cdna3.buffer_store(key, ptr=ckey_ptr, offsets=r * CAND + p, mask=mm)
            gl.amd.cdna3.buffer_store(idx, ptr=cidx_ptr, offsets=r * CAND + p, mask=mm)


@triton.jit
def _topk_select(ckey_ptr, cidx_ptr, ctr_ptr, lengths_ptr, page_ptr, out_ptr,
                 stride_p, CAND: tl.constexpr, TOPK: tl.constexpr,
                 CSTR: tl.constexpr):
    r = tl.program_id(0)
    n = tl.minimum(tl.load(ctr_ptr + r * CSTR + 1), CAND)
    k = tl.minimum(tl.load(lengths_ptr + r), TOPK)
    o = tl.arange(0, CAND)
    m = o < n
    cb = r.to(tl.int64) * CAND
    key = tl.load(ckey_ptr + cb + o, mask=m, other=-2147483648)
    thr = 0
    if tl.sum((key >= 0).to(tl.int32), 0) < k:
        thr = -2147483648
    for i in tl.static_range(0, 31):
        cand = thr + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= k:
            thr = cand
    gt = m & (key > thr)
    gti = gt.to(tl.int32)
    eq = m & (key == thr)
    eqi = eq.to(tl.int32)
    # one packed 8192-wide scan carries both ranks (each field < CAND <= 65536)
    cs = tl.cumsum(gti + (eqi * 65536), 0)
    rank_gt = (cs & 65535) - gti
    rank_eq = (cs // 65536) - eqi
    above = tl.sum(gti, 0)
    room = k - above
    take = gt | (eq & (rank_eq < room))
    n_take = tl.sum(take.to(tl.int32), 0)
    pos = rank_gt + tl.minimum(rank_eq, room)
    idx = tl.load(cidx_ptr + cb + o, mask=take, other=0)
    slot = tl.load(page_ptr + r.to(tl.int64) * stride_p + idx, mask=take, other=0)
    ot = tl.arange(0, TOPK)
    tl.store(out_ptr + r * TOPK + ot, tl.full((TOPK,), -1, tl.int32),
             mask=ot >= n_take)
    tl.store(out_ptr + r * TOPK + pos, slot, mask=take)


def topk_transform_triton(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table_1: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    rows = logits.shape[0]
    dev = logits.device
    out = torch.empty((rows, topk), dtype=torch.int32, device=dev)
    ctr = torch.empty((rows, _CSTR), dtype=torch.int32, device=dev)
    pref = torch.empty((rows + 1,), dtype=torch.int32, device=dev)
    ckey = torch.empty((rows, _TOPK_CAND), dtype=torch.int32, device=dev)
    cidx = torch.empty((rows, _TOPK_CAND), dtype=torch.int32, device=dev)
    li = logits.view(torch.int32)
    sl = li.stride(0)
    assert rows <= _RMAX
    assert rows * sl * 4 < 2**31 and rows * _TOPK_CAND * 4 < 2**31  # 32-bit buffer offsets
    _topk_est[(rows,)](li, lengths, pref, ctr, sl, rows, CH=_SMP_CH,
                       NCH=_SMP_NCH, TOPK=topk, OVER=_TOPK_OVER, T=_TOPK_T,
                       RMAX=_RMAX, CSTR=_CSTR, num_warps=8)
    _topk_collect[(_TOPK_GRID,)](li, lengths, pref, ctr, ckey, cidx, rows, sl,
                                 T=_TOPK_T, CAND=_TOPK_CAND, RMAX=_RMAX,
                                 GRID=_TOPK_GRID, CSTR=_CSTR, NW=_COLLECT_NW,
                                 num_warps=_COLLECT_NW)
    _topk_select[(rows,)](ckey, cidx, ctr, lengths, page_table_1, out,
                          page_table_1.stride(0), CAND=_TOPK_CAND, TOPK=topk,
                          CSTR=_CSTR, num_warps=8)
    return out


def topk_transform_fused(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    page_table_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """DSATopKBackend.topk_transform, SGL_KERNEL + PAGED + SGLANG_OPT_USE_TOPK_V2=false
    (sglang srt/layers/attention/dsa/dsa_topk_backend.py:96-212; fast_topk_transform_fused at :204)."""
    if _TOPK_IMPL == "triton":
        return topk_transform_triton(logits, lengths, page_table_1, topk)
    import sgl_kernel  # noqa: F401  (registers torch.ops.sgl_kernel.*)

    rows = logits.shape[0]
    out = logits.new_empty((rows, topk), dtype=torch.int32)
    op = torch.ops.sgl_kernel.fast_topk_transform_fused
    chunk = _TOPK_ROW_CHUNK
    if chunk <= 0 or rows <= chunk:
        op(logits, lengths, out, page_table_1, cu_seqlens_q, None)
        return out
    # The compiled coop_topk kernel has a row-count dispatch cliff: above ~60 rows it
    # falls off the multi-block path and nearly doubles in cost for identical data
    # volume (bs12: 93.5 us at 60 rows vs 179.8 us at 72). Calling it on contiguous
    # row-chunks keeps every launch on the fast path; semantics are unchanged because
    # rows are independent (paged mode, row_starts=None).
    nchunks = (rows + chunk - 1) // chunk
    bounds = [(rows * i) // nchunks for i in range(nchunks + 1)]
    for i in range(nchunks):
        s, e = bounds[i], bounds[i + 1]
        n = e - s
        op(logits[s:e], lengths[s:e], out[s:e], page_table_1[s:e],
           cu_seqlens_q[: n + 1], None)
    return out


# ===========================================================================
# Public entry point
# ===========================================================================
PAGE_SIZE = 64


def dsa_indexer_topk_paged(
    q_fp8: torch.Tensor,
    kv_cache_fp8: torch.Tensor,
    weights: torch.Tensor,
    seqlens_expanded: torch.Tensor,
    real_page_table: torch.Tensor,
    page_table_1: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    index_topk: int = 2048,
) -> torch.Tensor:
    """Top-k(2048) DSA indexer for TARGET_VERIFY rows (one query token per row).

    Args:
      q_fp8:            [rows, 32, 128] float8_e4m3fn index queries (act_quant'ed, ue8m0 scale folded into weights)
      kv_cache_fp8:     [num_pages, 64, 1, 132] uint8 -- per page: 64x128 fp8 K in the aiter MFMA-16x16
                        *preshuffled* layout, then 64 fp32 scales (indexer_k_quant_and_cache(preshuffle=True))
      weights:          [rows, 32] float32 per-head gate (already * n_heads**-0.5 * softmax_scale * q_scale)
      seqlens_expanded: [rows] int32 causal length of each row (L+1 .. L+6 for a request with prefix L)
      real_page_table:  [rows, W64] int32 page ids (page_size 64), row-expanded
      page_table_1:     [rows, W1] int32 token slots (page_size 1), row-expanded; W64 = ceil(W1 / 64)
      cu_seqlens_q:     [rows + 1] int32 = arange(rows + 1)
    Returns:
      [rows, index_topk] int32 global token slots of the top-k index scores, -1 padded.
    """
    max_seq_len = real_page_table.shape[1] * PAGE_SIZE
    kv_cache_fp8 = kv_cache_fp8.view(kv_cache_fp8.shape[0], PAGE_SIZE, 1, 132)
    logits = aiter_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        seqlens_expanded,
        real_page_table,
        max_seq_len,
        preshuffle=True,
        kv_block_size=PAGE_SIZE,
    )
    # _mask_init_and_local_tokens: no-op (GLM-5.2 index_init_tokens = index_local_tokens = 0).
    return topk_transform_fused(logits, seqlens_expanded, page_table_1, cu_seqlens_q, index_topk)
