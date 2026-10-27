"""Single-pass split-dim sparse-MLA prefill kernel for KV pools larger than 2 GB (FP8 KV, D_V = 512, 4 groups of 128).

Triton's AMD backend emits buffer loads (32-bit offsets) only for pointer arguments whose storage is <= 2**31 - 1 bytes;
a larger KV pool falls back to 64-bit global loads and the production kernel runs about 1.4x slower. Here the pool is
passed as up to MAX_CHUNKS views of <= 2 GB each (objects exposing data_ptr() and ptr_range()), every KV tile is
loaded once per view with a mask selecting that view's rows (masked-off lanes read 0), and the byte tiles are OR-ed.
Memory traffic is unchanged; only the load instruction count grows with the number of views.
Same math and loop order as `_sparse_mla_fwd_split_dim_kernel`."""

import torch
import triton
import triton.language as tl

MAX_CHUNKS = 4
_ROW_BYTES_LIMIT = 2**31 - 1


class _PoolView:
    """A pointer argument covering rows [start, start + rows) of a [num_pages, 1, dim] uint8 view of the pool."""

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


@triton.jit
def _ld(k0, k1, k2, k3, off, rmask, cid, NUM_CHUNKS: tl.constexpr):
    x = tl.load(k0 + off, mask=rmask & (cid == 0), other=0)
    if NUM_CHUNKS > 1:
        x |= tl.load(k1 + off, mask=rmask & (cid == 1), other=0)
    if NUM_CHUNKS > 2:
        x |= tl.load(k2 + off, mask=rmask & (cid == 2), other=0)
    if NUM_CHUNKS > 3:
        x |= tl.load(k3 + off, mask=rmask & (cid == 3), other=0)
    return x


@triton.jit
def sparse_mla_fwd_chunked_kernel(
    q_nope_ptr, q_rope_ptr, kv0, kv1, kv2, kv3, idx_ptr, o_ptr,
    qk_scale, fp8_max, topk, rows_per_chunk,
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
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        page = tl.where(valid, idx, 0)
        cid = page // rows_per_chunk
        loc = page - cid * rows_per_chunk
        rbase = loc[:, None] * DIM
        rmask = valid[:, None]
        c2 = cid[:, None]
        kv0_t = _ld(kv0, kv1, kv2, kv3, rbase + g[None, :], rmask, c2, NUM_CHUNKS).to(KV_DTYPE, bitcast=True)
        kv1_t = _ld(kv0, kv1, kv2, kv3, rbase + (G + g)[None, :], rmask, c2, NUM_CHUNKS).to(KV_DTYPE, bitcast=True)
        kv2_t = _ld(kv0, kv1, kv2, kv3, rbase + (2 * G + g)[None, :], rmask, c2, NUM_CHUNKS).to(KV_DTYPE, bitcast=True)
        kv3_t = _ld(kv0, kv1, kv2, kv3, rbase + (3 * G + g)[None, :], rmask, c2, NUM_CHUNKS).to(KV_DTYPE, bitcast=True)
        kv_tail = _ld(kv0, kv1, kv2, kv3, rbase + (D_V + dt)[None, :], rmask, c2, NUM_CHUNKS).to(KV_DTYPE, bitcast=True)

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


class _ClaimedRange:
    """Pointer argument for the whole pool that declares a <= 2 GB range, so Triton specialises it for buffer loads.
    Only valid if every load goes through a per-tile scalar base whose offsets stay below 2 GB (tile-base kernel)."""

    def __init__(self, t: torch.Tensor):
        self._ptr = t.data_ptr()
        self.dtype = t.dtype
        self.device = t.device

    def data_ptr(self):
        return self._ptr

    def ptr_range(self):
        return _ROW_BYTES_LIMIT


@triton.jit
def sparse_mla_fwd_tilebase_kernel(
    q_nope_ptr, q_rope_ptr, kv_ptr, idx_ptr, o_ptr,
    qk_scale, fp8_max, topk,
    H: tl.constexpr, DIM: tl.constexpr, D_V: tl.constexpr, D_TAIL: tl.constexpr,
    STRIDE_QN_T: tl.constexpr, STRIDE_QN_H: tl.constexpr, STRIDE_QR_T: tl.constexpr, STRIDE_QR_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    G: tl.constexpr = 128
    s_i = tl.program_id(0)
    h = tl.arange(0, H)
    dt = tl.arange(0, D_TAIL)
    g = tl.arange(0, G)
    in_t = kv_ptr.dtype.element_ty

    q_row = q_nope_ptr + s_i * STRIDE_QN_T + h[:, None] * STRIDE_QN_H
    q0 = tl.load(q_row + g[None, :]).to(in_t)
    q1 = tl.load(q_row + (G + g)[None, :]).to(in_t)
    q2 = tl.load(q_row + (2 * G + g)[None, :]).to(in_t)
    q3 = tl.load(q_row + (3 * G + g)[None, :]).to(in_t)
    q_tail = tl.load(q_rope_ptr + s_i * STRIDE_QR_T + h[:, None] * STRIDE_QR_H + dt[None, :]).to(in_t)

    neg_large = -3.4028234663852886e38
    m_i = tl.full([H], neg_large, tl.float32)
    l_i = tl.zeros([H], tl.float32)
    acc0 = tl.zeros([H, G], tl.float32)
    acc1 = tl.zeros([H, G], tl.float32)
    acc2 = tl.zeros([H, G], tl.float32)
    acc3 = tl.zeros([H, G], tl.float32)
    p_dot_scale = 1.0 / fp8_max
    n = tl.arange(0, BLOCK_N)
    for k0 in range(0, topk, BLOCK_N):
        kmask = (k0 + n) < topk
        idx = tl.load(idx_ptr + s_i * topk + k0 + n, mask=kmask, other=-1)
        valid = (idx >= 0) & kmask
        rmin = tl.min(tl.where(valid, idx, 2147483647))
        rmin = tl.where(rmin == 2147483647, 0, rmin)
        rel = tl.where(valid, idx - rmin, 0)
        kbase = (kv_ptr + rmin.to(tl.int64) * DIM) + rel[:, None] * DIM
        kv0 = tl.load(kbase + g[None, :], mask=valid[:, None], other=0.0)
        kv1 = tl.load(kbase + (G + g)[None, :], mask=valid[:, None], other=0.0)
        kv2 = tl.load(kbase + (2 * G + g)[None, :], mask=valid[:, None], other=0.0)
        kv3 = tl.load(kbase + (3 * G + g)[None, :], mask=valid[:, None], other=0.0)
        kv_tail = tl.load(kbase + (D_V + dt)[None, :], mask=valid[:, None], other=0.0)

        qk = tl.dot(q0, tl.trans(kv0))
        qk += tl.dot(q1, tl.trans(kv1))
        qk += tl.dot(q2, tl.trans(kv2))
        qk += tl.dot(q3, tl.trans(kv3))
        qk += tl.dot(q_tail, tl.trans(kv_tail))
        qk = qk * qk_scale
        qk = tl.where(valid[None, :], qk, neg_large)

        m_block = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        p_dot = (p * fp8_max).to(in_t)
        acc0 = acc0 * alpha[:, None] + tl.dot(p_dot, kv0).to(tl.float32) * p_dot_scale
        acc1 = acc1 * alpha[:, None] + tl.dot(p_dot, kv1).to(tl.float32) * p_dot_scale
        acc2 = acc2 * alpha[:, None] + tl.dot(p_dot, kv2).to(tl.float32) * p_dot_scale
        acc3 = acc3 * alpha[:, None] + tl.dot(p_dot, kv3).to(tl.float32) * p_dot_scale
        m_i = m_new

    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    inv_l = 1.0 / l_safe
    o_base = o_ptr + s_i * H * D_V + h[:, None] * D_V
    tl.store(o_base + g[None, :], (acc0 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (G + g)[None, :], (acc1 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (2 * G + g)[None, :], (acc2 * inv_l[:, None]).to(o_ptr.dtype.element_ty))
    tl.store(o_base + (3 * G + g)[None, :], (acc3 * inv_l[:, None]).to(o_ptr.dtype.element_ty))


def sparse_mla_fwd_tilebase(q_nope, q_rope, kv, idx_sorted, qk_scale, fp8_max, block_n=32, num_warps=4, num_stages=2,
                            claim=True):
    """idx_sorted: [seq, topk] int32 with every BLOCK_N-row tile spanning < 2 GB of the pool (e.g. rows sorted)."""
    seq, H, d_v = q_nope.shape
    topk = idx_sorted.shape[-1]
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    sparse_mla_fwd_tilebase_kernel[(seq,)](
        q_nope, q_rope, _ClaimedRange(kv) if claim else kv, idx_sorted, out, qk_scale, fp8_max, topk,
        H=H, DIM=kv.shape[-1], D_V=d_v, D_TAIL=q_rope.shape[-1],
        STRIDE_QN_T=q_nope.stride(0), STRIDE_QN_H=q_nope.stride(1),
        STRIDE_QR_T=q_rope.stride(0), STRIDE_QR_H=q_rope.stride(1),
        BLOCK_N=block_n, num_warps=num_warps, num_stages=num_stages)
    return out


_TL_FP8 = {torch.float8_e4m3fn: tl.float8e4nv, torch.float8_e4m3fnuz: tl.float8e4b8}


def sparse_mla_fwd_chunked(q_nope, q_rope, kv, idx_flat, qk_scale, fp8_max, block_n=32, num_warps=4, num_stages=2):
    """q_nope [seq, H, 512], q_rope [seq, H, 64] (FP8, unit last stride), kv [pages, 1, 576] FP8, idx_flat [seq, topk] int32."""
    seq, H, d_v = q_nope.shape
    assert d_v == 512 and kv.dtype in _TL_FP8
    topk = idx_flat.shape[-1]
    views, n_chunks, rows_per = pool_views(kv.view(torch.uint8))
    out = torch.empty(seq, H, d_v, device=q_nope.device, dtype=torch.bfloat16)
    sparse_mla_fwd_chunked_kernel[(seq,)](
        q_nope, q_rope, *views, idx_flat, out, qk_scale, fp8_max, topk, rows_per,
        H=H, DIM=kv.shape[-1], D_V=d_v, D_TAIL=q_rope.shape[-1],
        STRIDE_QN_T=q_nope.stride(0), STRIDE_QN_H=q_nope.stride(1),
        STRIDE_QR_T=q_rope.stride(0), STRIDE_QR_H=q_rope.stride(1),
        NUM_CHUNKS=n_chunks, BLOCK_N=block_n, KV_DTYPE=_TL_FP8[kv.dtype],
        num_warps=num_warps, num_stages=num_stages)
    return out
