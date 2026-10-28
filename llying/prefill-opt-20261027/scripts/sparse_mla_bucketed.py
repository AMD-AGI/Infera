"""Single-pass split-dim sparse-MLA prefill kernel for KV pools larger than 2 GB, one buffer load per KV tile.

Triton's AMD backend emits buffer loads (32-bit offsets) only for pointer arguments whose storage or ptr_range() is
<= 2**31 - 1 bytes; the production pool (6.2 GB per layer) falls back to 64-bit global loads, 5.12 vs 3.71 ms.
Here each query row's top-k indices are bucketed by 2 GB segment of the pool and every bucket is padded to BLOCK_N
(bucket_indices), so every KV tile lies in one segment; the kernel reads the tile's segment id and loads through that
segment's <= 2 GB view under a scalar branch. Attention is order-independent, so only the accumulation order changes.
Same math as `_sparse_mla_fwd_split_dim_kernel` (FP8 dots, exp2 online softmax)."""

import torch
import triton
import triton.language as tl

MAX_CHUNKS = 4
_ROW_BYTES_LIMIT = 2**31 - 1


class _PoolView:
    """A pointer argument covering rows [start, start + rows) of a [num_rows, 1, dim] uint8 view of the pool."""

    def __init__(self, t: torch.Tensor, start: int, rows: int):
        row_bytes = t.stride(0) * t.element_size()
        self._ptr = t.data_ptr() + start * row_bytes
        self._range = rows * row_bytes
        self.dtype = torch.uint8
        self.device = t.device

    def data_ptr(self):
        return self._ptr

    def ptr_range(self):
        return self._range


def pool_views(kv: torch.Tensor):
    row_bytes = kv.stride(0) * kv.element_size()
    rows_per = _ROW_BYTES_LIMIT // row_bytes
    n = kv.shape[0]
    views = [_PoolView(kv, s, min(rows_per, n - s)) for s in range(0, n, rows_per)]
    assert len(views) <= MAX_CHUNKS, f"KV pool of {n} rows needs {len(views)} views > {MAX_CHUNKS}"
    return views + [views[0]] * (MAX_CHUNKS - len(views)), len(views), rows_per


def bucket_indices(idx: torch.Tensor, rows_per_chunk: int, n_chunks: int, block_n: int):
    """idx [S, K] int32 pool rows (-1 = none) -> (local [S, T * block_n] int32 rows within the segment, -1 = pad;
    tile_cid [S, T] int32 segment of each tile, -1 = empty tile)."""
    S, K = idx.shape
    T = (K + n_chunks * (block_n - 1) + block_n - 1) // block_n
    L = T * block_n
    big = torch.iinfo(torch.int32).max
    srt = torch.sort(torch.where(idx >= 0, idx, big), dim=1).values
    valid = srt != big
    cid = torch.where(valid, srt // rows_per_chunk, n_chunks).to(torch.int64)
    ar = torch.arange(n_chunks, device=idx.device)
    counts = (cid.unsqueeze(-1) == ar).sum(1)                                     # [S, C]
    padded = (counts + block_n - 1) // block_n * block_n
    start_pad = torch.cumsum(padded, 1) - padded                                  # [S, C]
    start_srt = torch.cumsum(counts, 1) - counts
    c_cl = cid.clamp(max=n_chunks - 1)
    pos = torch.arange(K, device=idx.device).expand(S, K)
    newpos = torch.gather(start_pad, 1, c_cl) + pos - torch.gather(start_srt, 1, c_cl)
    newpos = torch.where(valid, newpos, L)
    local = (srt - c_cl.to(torch.int32) * rows_per_chunk).to(torch.int32)
    out = torch.full((S, L + 1), -1, dtype=torch.int32, device=idx.device)
    out.scatter_(1, newpos, torch.where(valid, local, -1))
    tstart = (torch.arange(T, device=idx.device) * block_n).expand(S, T)
    end_pad = start_pad + padded
    tile_cid = torch.full((S, T), -1, dtype=torch.int32, device=idx.device)
    for c in range(n_chunks):
        inside = (tstart >= start_pad[:, c:c + 1]) & (tstart < end_pad[:, c:c + 1])
        tile_cid = torch.where(inside, c, tile_cid)
    return out[:, :L].contiguous(), tile_cid.contiguous(), T


@triton.jit
def _ld_tile(k0, k1, k2, k3, rbase, g, dt, rmask, cid, G: tl.constexpr, D_V: tl.constexpr, NUM_CHUNKS: tl.constexpr):
    if cid == 0:
        a0 = tl.load(k0 + rbase + g[None, :], mask=rmask, other=0)
        a1 = tl.load(k0 + rbase + (G + g)[None, :], mask=rmask, other=0)
        a2 = tl.load(k0 + rbase + (2 * G + g)[None, :], mask=rmask, other=0)
        a3 = tl.load(k0 + rbase + (3 * G + g)[None, :], mask=rmask, other=0)
        at = tl.load(k0 + rbase + (D_V + dt)[None, :], mask=rmask, other=0)
    elif cid == 1:
        a0 = tl.load(k1 + rbase + g[None, :], mask=rmask, other=0)
        a1 = tl.load(k1 + rbase + (G + g)[None, :], mask=rmask, other=0)
        a2 = tl.load(k1 + rbase + (2 * G + g)[None, :], mask=rmask, other=0)
        a3 = tl.load(k1 + rbase + (3 * G + g)[None, :], mask=rmask, other=0)
        at = tl.load(k1 + rbase + (D_V + dt)[None, :], mask=rmask, other=0)
    elif cid == 2:
        a0 = tl.load(k2 + rbase + g[None, :], mask=rmask, other=0)
        a1 = tl.load(k2 + rbase + (G + g)[None, :], mask=rmask, other=0)
        a2 = tl.load(k2 + rbase + (2 * G + g)[None, :], mask=rmask, other=0)
        a3 = tl.load(k2 + rbase + (3 * G + g)[None, :], mask=rmask, other=0)
        at = tl.load(k2 + rbase + (D_V + dt)[None, :], mask=rmask, other=0)
    else:
        a0 = tl.load(k3 + rbase + g[None, :], mask=rmask, other=0)
        a1 = tl.load(k3 + rbase + (G + g)[None, :], mask=rmask, other=0)
        a2 = tl.load(k3 + rbase + (2 * G + g)[None, :], mask=rmask, other=0)
        a3 = tl.load(k3 + rbase + (3 * G + g)[None, :], mask=rmask, other=0)
        at = tl.load(k3 + rbase + (D_V + dt)[None, :], mask=rmask, other=0)
    return a0, a1, a2, a3, at


@triton.jit
def sparse_mla_fwd_bucketed_kernel(
    q_nope_ptr, q_rope_ptr, kv0, kv1, kv2, kv3, idx_ptr, cid_ptr, o_ptr,
    qk_scale, fp8_max, n_tiles,
    H: tl.constexpr, DIM: tl.constexpr, D_V: tl.constexpr, D_TAIL: tl.constexpr,
    STRIDE_QN_T: tl.constexpr, STRIDE_QN_H: tl.constexpr, STRIDE_QR_T: tl.constexpr, STRIDE_QR_H: tl.constexpr,
    NUM_CHUNKS: tl.constexpr, BLOCK_N: tl.constexpr, KV_DTYPE: tl.constexpr,
):
    G: tl.constexpr = 128
    s_i = tl.program_id(0)
    h = tl.arange(0, H)
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, G)

    q_row = q_nope_ptr + s_i * STRIDE_QN_T + h[:, None] * STRIDE_QN_H
    q0 = tl.load(q_row + g[None, :]).to(KV_DTYPE)
    q1 = tl.load(q_row + (G + g)[None, :]).to(KV_DTYPE)
    q2 = tl.load(q_row + (2 * G + g)[None, :]).to(KV_DTYPE)
    q3 = tl.load(q_row + (3 * G + g)[None, :]).to(KV_DTYPE)
    q_tail = tl.load(q_rope_ptr + s_i * STRIDE_QR_T + h[:, None] * STRIDE_QR_H + dt[None, :]).to(KV_DTYPE)

    neg_large = -3.4028234663852886e38
    m_i = tl.full([H], neg_large, tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc0 = tl.zeros([H, G], tl.float32)
    acc1 = tl.zeros([H, G], tl.float32)
    acc2 = tl.zeros([H, G], tl.float32)
    acc3 = tl.zeros([H, G], tl.float32)
    p_dot_scale = 1.0 / fp8_max
    n = tl.arange(0, BLOCK_N)
    L = n_tiles * BLOCK_N
    for t in range(0, n_tiles):
        cid = tl.load(cid_ptr + s_i * n_tiles + t)
        if cid >= 0:
            idx = tl.load(idx_ptr + s_i * L + t * BLOCK_N + n)
            valid = idx >= 0
            rbase = tl.where(valid, idx, 0)[:, None] * DIM
            rmask = valid[:, None]
            a0, a1, a2, a3, at = _ld_tile(kv0, kv1, kv2, kv3, rbase, g, dt, rmask, cid, G, D_V, NUM_CHUNKS)
            kv0_t = a0.to(KV_DTYPE, bitcast=True)
            kv1_t = a1.to(KV_DTYPE, bitcast=True)
            kv2_t = a2.to(KV_DTYPE, bitcast=True)
            kv3_t = a3.to(KV_DTYPE, bitcast=True)
            kv_tail = at.to(KV_DTYPE, bitcast=True)

            qk = tl.dot(q0, tl.trans(kv0_t))
            qk += tl.dot(q1, tl.trans(kv1_t))
            qk += tl.dot(q2, tl.trans(kv2_t))
            qk += tl.dot(q3, tl.trans(kv3_t))
            qk += tl.dot(q_tail, tl.trans(kv_tail))
            qk = qk * qk_scale
            qk = tl.where(valid[None, :], qk, neg_large)

            m_block = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            p_dot = (p * fp8_max).to(KV_DTYPE)
            acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0_t).to(tl.float32) * p_dot_scale
            acc1 = acc1 * alpha[:, None] + tl.dot(p_dot, kv1_t).to(tl.float32) * p_dot_scale
            acc2 = acc2 * alpha[:, None] + tl.dot(p_dot, kv2_t).to(tl.float32) * p_dot_scale
            acc3 = acc3 * alpha[:, None] + tl.dot(p_dot, kv3_t).to(tl.float32) * p_dot_scale
            m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe
    o_base = o_ptr + s_i * H * D_V + h[:, None] * D_V
    tl.store(o_base + g[None, :], (acc0 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (G + g)[None, :], (acc1 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (2 * G + g)[None, :], (acc2 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (3 * G + g)[None, :], (acc3 * inv_l[:, None]).to(o_ptr.dtype.element_ty))


@triton.jit
def _seg_loop(kvc, idx_ptr, s_i, L, t0, t1, q0, q1, q2, q3, q_tail, m_i, l_i, acc0, acc1, acc2, acc3,
              qk_scale, fp8_max, n, g, dt, G: tl.constexpr, DIM: tl.constexpr, D_V: tl.constexpr,
              BLOCK_N: tl.constexpr, KV_DTYPE: tl.constexpr):
    neg_large = -3.4028234663852886e38
    p_dot_scale = 1.0 / fp8_max
    for t in range(t0, t1):
        idx = tl.load(idx_ptr + s_i * L + t * BLOCK_N + n)
        valid = idx >= 0
        rbase = tl.where(valid, idx, 0)[:, None] * DIM
        rmask = valid[:, None]
        kv0_t = tl.load(kvc + rbase + g[None, :], mask=rmask, other=0).to(KV_DTYPE, bitcast=True)
        kv1_t = tl.load(kvc + rbase + (G + g)[None, :], mask=rmask, other=0).to(KV_DTYPE, bitcast=True)
        kv2_t = tl.load(kvc + rbase + (2 * G + g)[None, :], mask=rmask, other=0).to(KV_DTYPE, bitcast=True)
        kv3_t = tl.load(kvc + rbase + (3 * G + g)[None, :], mask=rmask, other=0).to(KV_DTYPE, bitcast=True)
        kv_tail = tl.load(kvc + rbase + (D_V + dt)[None, :], mask=rmask, other=0).to(KV_DTYPE, bitcast=True)
        qk = tl.dot(q0, tl.trans(kv0_t))
        qk += tl.dot(q1, tl.trans(kv1_t))
        qk += tl.dot(q2, tl.trans(kv2_t))
        qk += tl.dot(q3, tl.trans(kv3_t))
        qk += tl.dot(q_tail, tl.trans(kv_tail))
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, neg_large)
        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_dot = (p * fp8_max).to(KV_DTYPE)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0_t).to(tl.float32) * p_dot_scale
        acc1 = acc1 * alpha[:, None] + tl.dot(p_dot, kv1_t).to(tl.float32) * p_dot_scale
        acc2 = acc2 * alpha[:, None] + tl.dot(p_dot, kv2_t).to(tl.float32) * p_dot_scale
        acc3 = acc3 * alpha[:, None] + tl.dot(p_dot, kv3_t).to(tl.float32) * p_dot_scale
        m_i = m_new
    return m_i, l_i, acc0, acc1, acc2, acc3


@triton.jit
def sparse_mla_fwd_seg_kernel(
    q_nope_ptr, q_rope_ptr, kv0, kv1, kv2, kv3, idx_ptr, seg_ptr, o_ptr,
    qk_scale, fp8_max, n_tiles,
    H: tl.constexpr, DIM: tl.constexpr, D_V: tl.constexpr, D_TAIL: tl.constexpr,
    STRIDE_QN_T: tl.constexpr, STRIDE_QN_H: tl.constexpr, STRIDE_QR_T: tl.constexpr, STRIDE_QR_H: tl.constexpr,
    NUM_CHUNKS: tl.constexpr, BLOCK_N: tl.constexpr, KV_DTYPE: tl.constexpr,
):
    """seg_ptr [S, NUM_CHUNKS, 2] int32: first tile and tile count of each segment's bucket in the row."""
    G: tl.constexpr = 128
    s_i = tl.program_id(0)
    h = tl.arange(0, H)
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, G)
    q_row = q_nope_ptr + s_i * STRIDE_QN_T + h[:, None] * STRIDE_QN_H
    q0 = tl.load(q_row + g[None, :]).to(KV_DTYPE)
    q1 = tl.load(q_row + (G + g)[None, :]).to(KV_DTYPE)
    q2 = tl.load(q_row + (2 * G + g)[None, :]).to(KV_DTYPE)
    q3 = tl.load(q_row + (3 * G + g)[None, :]).to(KV_DTYPE)
    q_tail = tl.load(q_rope_ptr + s_i * STRIDE_QR_T + h[:, None] * STRIDE_QR_H + dt[None, :]).to(KV_DTYPE)
    m_i = tl.full([H], -3.4028234663852886e38, tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc0 = tl.zeros([H, G], tl.float32)
    acc1 = tl.zeros([H, G], tl.float32)
    acc2 = tl.zeros([H, G], tl.float32)
    acc3 = tl.zeros([H, G], tl.float32)
    n = tl.arange(0, BLOCK_N)
    L = n_tiles * BLOCK_N
    for c in tl.static_range(NUM_CHUNKS):
        if c == 0:
            kvc = kv0
        elif c == 1:
            kvc = kv1
        elif c == 2:
            kvc = kv2
        else:
            kvc = kv3
        t0 = tl.load(seg_ptr + (s_i * NUM_CHUNKS + c) * 2)
        t1 = t0 + tl.load(seg_ptr + (s_i * NUM_CHUNKS + c) * 2 + 1)
        m_i, l_i, acc0, acc1, acc2, acc3 = _seg_loop(
            kvc, idx_ptr, s_i, L, t0, t1, q0, q1, q2, q3, q_tail, m_i, l_i, acc0, acc1, acc2, acc3,
            qk_scale, fp8_max, n, g, dt, G, DIM, D_V, BLOCK_N, KV_DTYPE)
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe
    o_base = o_ptr + s_i * H * D_V + h[:, None] * D_V
    tl.store(o_base + g[None, :], (acc0 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (G + g)[None, :], (acc1 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (2 * G + g)[None, :], (acc2 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (3 * G + g)[None, :], (acc3 * inv_l[:, None]).to(o_ptr.dtype.element_ty))


def seg_ranges(tile_cid: torch.Tensor, n_chunks: int):
    """[S, T] tile segment ids -> [S, n_chunks, 2] int32 (first tile, tile count); buckets are contiguous in order."""
    ar = torch.arange(n_chunks, device=tile_cid.device, dtype=tile_cid.dtype)
    cnt = (tile_cid.unsqueeze(-1) == ar).sum(1).to(torch.int32)
    first = (torch.cumsum(cnt, 1) - cnt).to(torch.int32)
    return torch.stack([first, cnt], -1).contiguous()


def sparse_mla_fwd_seg(q_nope, q_rope, kv, idx_flat, qk_scale, fp8_max, block_n=64, num_warps=4, num_stages=2,
                       bucketed=None):
    S, H, DV = q_nope.shape
    views, n_chunks, rows_per = pool_views(kv.view(torch.uint8))
    if bucketed is None:
        local, tile_cid, T = bucket_indices(idx_flat, rows_per, n_chunks, block_n)
        bucketed = (local, seg_ranges(tile_cid, n_chunks), T)
    local, seg, T = bucketed
    out = torch.empty(S, H, DV, device=q_nope.device, dtype=torch.bfloat16)
    sparse_mla_fwd_seg_kernel[(S,)](
        q_nope, q_rope, *views, local, seg, out, qk_scale, fp8_max, T,
        H=H, DIM=kv.shape[-1], D_V=DV, D_TAIL=q_rope.shape[-1],
        STRIDE_QN_T=q_nope.stride(0), STRIDE_QN_H=q_nope.stride(1),
        STRIDE_QR_T=q_rope.stride(0), STRIDE_QR_H=q_rope.stride(1),
        NUM_CHUNKS=n_chunks, BLOCK_N=block_n, KV_DTYPE=_TL_FP8[kv.dtype],
        num_warps=num_warps, num_stages=num_stages)
    return out


_TL_FP8 = {torch.float8_e4m3fnuz: tl.float8e4b8, torch.float8_e4m3fn: tl.float8e4nv}


def sparse_mla_fwd_bucketed(q_nope, q_rope, kv, idx_flat, qk_scale, fp8_max, block_n=64, num_warps=4, num_stages=2,
                            bucketed=None):
    """q_nope [S, H, 512], q_rope [S, H, 64], kv [rows, 1, 576] FP8, idx_flat [S, K] int32 -> out [S, H, 512] bf16.
    bucketed: precomputed (local, tile_cid, T) from bucket_indices, to time the kernel alone."""
    S, H, DV = q_nope.shape
    views, n_chunks, rows_per = pool_views(kv.view(torch.uint8))
    if bucketed is None:
        bucketed = bucket_indices(idx_flat, rows_per, n_chunks, block_n)
    local, tile_cid, T = bucketed
    out = torch.empty(S, H, DV, device=q_nope.device, dtype=torch.bfloat16)
    sparse_mla_fwd_bucketed_kernel[(S,)](
        q_nope, q_rope, *views, local, tile_cid, out, qk_scale, fp8_max, T,
        H=H, DIM=kv.shape[-1], D_V=DV, D_TAIL=q_rope.shape[-1],
        STRIDE_QN_T=q_nope.stride(0), STRIDE_QN_H=q_nope.stride(1),
        STRIDE_QR_T=q_rope.stride(0), STRIDE_QR_H=q_rope.stride(1),
        NUM_CHUNKS=n_chunks, BLOCK_N=block_n, KV_DTYPE=_TL_FP8[kv.dtype],
        num_warps=num_warps, num_stages=num_stages)
    return out
