# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-pass top-k (<= 2048) for DSA indexer decode / MTP-verify rows
(ROCm, <= 128 rows): est (Triton) -> collect (Gluon) -> select (Triton).

Ported from sglang PR #47872 (``triton_topk_transform.py``). vLLM's decode
top-k returns token positions inside the row (the sparse MLA backend maps
them through the block table), so the page transform is optional here
(``TRANSFORM``), and the indices are written into the caller's buffer.

  est     per row, sample ~8192 logits (64 probes of 128) and bit-search the
          sample key that leaves ~OVER * topk elements above it in the full row
          -> a per-row threshold (rounded down, so it only adds candidates);
          program 0 also builds the chunk prefix.
  collect ONE pass over the logits on a fixed persistent grid of row-aligned
          chunks; every key >= threshold is appended to the row's candidate
          buffer (CAND entries).
  select  per row, the exact k-th largest key among the candidates, then
          exactly min(L, topk) slots through the page table.  If a row has fewer
          than min(L, topk) candidates or overflows CAND, it selects exactly from
          the whole row instead (slow, rare), so the result is always an exact
          top-k.

vpd changes (P4D4 GLM-5.2, 11-24): real indexer logits can have one value (0) on
hundreds of thousands of positions of a long row; the sampled threshold then
lands on it, collect overflows CAND and every such row took the whole-row path
(a11: select p50 4 ms, decode throughput halved). est now also finds the exact
t-th sample key and, if that value is expected more than _TIE_MIN times in the
row, collects only keys above it; select takes all of those and fills the rest
with tied elements in row order (exact top-k; ties are interchangeable).
That alone did not fix production (a14: select still on the slow path for most
calls): the logits are correlated along the row, so 64 contiguous 128-element
probes misjudge the threshold for a few % of rows, and one slow row makes the
whole call slow. Now 2048 probes of 4, OVER 4, CAND 16384 (scripts/sim_topk_est.py),
and the whole-row fallback uses 4 radix passes instead of a 31-pass bit search.

Output order is not defined, and when the k-th key has ties the members chosen
among the tied elements depend on candidate arrival order.  The launch shape
depends only on the row count (CUDA-graph safe).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_T = 8192  # logits per collect chunk
_GRID = 2048  # persistent collect programs
# vpd: 2048 short probes (was 64 x 128) and OVER 4 / CAND 16384 (was 2 / 8192). Real indexer logits are correlated
# along the row; with long contiguous probes the effective sample is small and the threshold misses by > 2x often
# (scripts/sim_topk_est.py: 18-56% of 700K rows outside [k, CAND] on AR / segment / hot-region rows, 0 of 1,000 with
# 2048 x 4, OVER 4, CAND 16384, candidate counts 2.2k..5.7k of the [k, CAND] window).
_CAND = 16384  # candidates per row
_OVER = 4  # candidate target, x topk
_SMP_CH = 4  # probe length
_SMP_NCH = 2048  # probes per row
MAX_ROWS = 128
_CSTR = 64  # int32 gap between rows' counters
_COLLECT_NW = 4
_FALLBACK_BLOCK = 4096
_TIE_MIN = _CAND // 4  # tie mode when the t-th sample key is expected > this many times in the row


@triton.jit
def _sum_combine(a, b):
    return a + b


@triton.jit
def _okey(b):
    # order-preserving int32 key of a float32 bit pattern
    return tl.where(b >= 0, b, (~b) ^ (-2147483648))


@triton.jit
def _topk_est(
    logits_ptr,
    lengths_ptr,
    pref_ptr,
    ctr_ptr,
    stride_l,
    rows,
    CH: tl.constexpr,
    NCH: tl.constexpr,
    TOPK: tl.constexpr,
    OVER: tl.constexpr,
    T: tl.constexpr,
    RMAX: tl.constexpr,
    CSTR: tl.constexpr,
    TIE_MIN: tl.constexpr,
):
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
    t = tl.maximum(1, tl.minimum((OVER * k * ns) // tl.maximum(L, 1), ns))
    thr = 0
    if tl.sum((key >= 0).to(tl.int32), 0) < t:
        thr = -2147483648
    # bits 30..15 only: rounding the sampled threshold down can only add
    # candidates. Sampling can still overestimate the full-row k-th key;
    # the selector handles that case with its exact underflow fallback.
    for i in tl.static_range(0, 16):
        cand = thr + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= t:
            thr = cand
    # Tie guard: refine to the exact t-th sample key kt. If that one value is expected to fill a large part of the
    # row (e.g. indexer logits that are exactly 0 wherever every head's ReLU is 0), collecting `>= thr` overflows
    # the candidate buffer and the selector falls back to its slow whole-row path. Then collect only `> kt` and let
    # the selector take the missing `== kt` elements from the row (any of them: tied elements are interchangeable).
    kt = thr
    for i in tl.static_range(16, 31):
        cand = kt + (1 << (30 - i))
        if tl.sum((key >= cand).to(tl.int32), 0) >= t:
            kt = cand
    n_eq = tl.sum((m & (key == kt)).to(tl.int32), 0)
    strict = ((n_eq.to(tl.int64) * L) // tl.maximum(ns, 1) > TIE_MIN) & (kt < 2147483647)
    tl.store(ctr_ptr + r * CSTR, tl.where(strict, kt + 1, thr))
    tl.store(ctr_ptr + r * CSTR + 2, kt)
    tl.store(ctr_ptr + r * CSTR + 3, strict.to(tl.int32))


@gluon.jit
def _topk_collect(
    logits_ptr,
    lengths_ptr,
    pref_ptr,
    ctr_ptr,
    ckey_ptr,
    cidx_ptr,
    rows,
    stride_l,
    T: gl.constexpr,
    CAND: gl.constexpr,
    RMAX: gl.constexpr,
    GRID: gl.constexpr,
    CSTR: gl.constexpr,
    NW: gl.constexpr,
):
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
        b = gl.amd.cdna3.buffer_load(
            ptr=logits_ptr, offsets=r * stride_l + idx, mask=(start + va) < L, other=-1
        )
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
            tb = gl.convert_layout(
                gl.associative_scan(cnt_b, 0, _sum_combine) - cnt_b,
                gl.SliceLayout(0, LRT),
            )
            s_vb = gl.convert_layout(s_v, L2)
            rv = gl.convert_layout(
                gl.associative_scan(s_vb, 0, _sum_combine) - s_vb, LRT
            )
            rk = gl.expand_dims(rv, 2) + (gl.associative_scan(si, 2, _sum_combine) - si)
            p = base + gl.expand_dims(gl.expand_dims(tb, 0), 2) + rk
            mm = (si != 0) & (p < CAND)
            gl.amd.cdna3.buffer_store(key, ptr=ckey_ptr, offsets=r * CAND + p, mask=mm)
            gl.amd.cdna3.buffer_store(idx, ptr=cidx_ptr, offsets=r * CAND + p, mask=mm)


@triton.jit
def _row_threshold(row_ptr, L, k, BLOCK: tl.constexpr):
    """Exact k-th largest key of a whole row: 4 radix passes of 8 bits over the row (was a 31-pass bit search,
    ~4 ms on a 700K row with one program). Works on u = key ^ INT_MIN, whose unsigned order is the key order."""
    bins = tl.arange(0, 512)
    zero = L * 0  # int32 scalars throughout: shifts into bit 31 wrap to the wanted pattern
    ff = zero + 255
    prefix = zero
    pmask = zero
    need = k
    for p in tl.static_range(0, 4):
        h = tl.zeros((512,), tl.int32)
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            u = _okey(tl.load(row_ptr + o, mask=o < L, other=-1)) ^ (-2147483648)
            hit = (o < L) & ((u & pmask) == prefix)
            d = tl.where(hit, (u >> (24 - 8 * p)) & ff, 511)
            h += tl.histogram(d, 512)
        h = tl.where(bins < 256, h, 0)
        ge = tl.sum(h, 0) - tl.cumsum(h, 0) + h  # elements in bins >= b
        b = tl.max(tl.where(ge >= need, bins, -1), 0)
        need -= tl.sum(tl.where(bins > b, h, 0), 0)
        prefix = prefix | (b.to(tl.int32) << (24 - 8 * p))
        pmask = pmask | (ff << (24 - 8 * p))
    return prefix ^ (-2147483648)


@triton.jit
def _topk_select(
    ckey_ptr,
    cidx_ptr,
    ctr_ptr,
    lengths_ptr,
    logits_ptr,
    stride_l,
    page_ptr,
    stride_p,
    out_ptr,
    stride_o,
    CAND: tl.constexpr,
    TOPK: tl.constexpr,
    CSTR: tl.constexpr,
    PAGE_SHIFT: tl.constexpr,
    BLOCK: tl.constexpr,
    TRANSFORM: tl.constexpr,
):
    r = tl.program_id(0)
    n_all = tl.load(ctr_ptr + r * CSTR + 1)
    L = tl.load(lengths_ptr + r)
    k = tl.minimum(L, TOPK)
    pt = page_ptr + r.to(tl.int64) * stride_p
    out_cols = tl.arange(0, TOPK)
    strict = tl.load(ctr_ptr + r * CSTR + 3)
    tie = tl.load(ctr_ptr + r * CSTR + 2)
    zero = n_all * 0
    tie_ok = zero
    if (strict != 0) & (n_all < k):
        # Tie mode (see _topk_est): every candidate (> tie) is in the top-k; the other k - n_all slots go to
        # elements equal to the tie value, taken in row order until enough are found (usually the first block).
        oc = tl.arange(0, CAND)
        mc = oc < n_all
        cb0 = r.to(tl.int64) * CAND
        idc = tl.load(cidx_ptr + cb0 + oc, mask=mc, other=0)
        if TRANSFORM:
            sc = (
                tl.load(pt + (idc >> PAGE_SHIFT), mask=mc, other=0) << PAGE_SHIFT
            ) + (idc & ((1 << PAGE_SHIFT) - 1))
        else:
            sc = idc
        tl.store(out_ptr + r * stride_o + oc, sc, mask=mc)
        need = k - n_all
        rowt = logits_ptr + r.to(tl.int64) * stride_l
        filled = zero
        s = zero
        while (filled < need) & (s < L):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(rowt + o, mask=o < L, other=-1))
            eq = (o < L) & (key == tie)
            eqi = eq.to(tl.int32)
            rank = filled + tl.cumsum(eqi, 0) - eqi
            take = eq & (rank < need)
            if TRANSFORM:
                slot = (
                    tl.load(pt + (o >> PAGE_SHIFT), mask=take, other=0) << PAGE_SHIFT
                ) + (o & ((1 << PAGE_SHIFT) - 1))
            else:
                slot = o
            tl.store(out_ptr + r * stride_o + n_all + rank, slot, mask=take)
            filled += tl.sum(take.to(tl.int32), 0)
            s += BLOCK
        tie_ok = (filled == need).to(tl.int32)
        tl.store(
            out_ptr + r * stride_o + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=(out_cols >= k) & (tie_ok != 0),
        )
    if ((n_all < k) & (tie_ok == 0)) | (n_all > CAND):
        # The sampled threshold was too high (underflow) or too low (overflow):
        # select exactly from the whole row instead. Rare and slow.
        row = logits_ptr + r.to(tl.int64) * stride_l
        thr = _row_threshold(row, L, k, BLOCK)
        base = 0
        above = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(row + o, mask=o < L, other=-1))
            above += tl.sum(((o < L) & (key > thr)).to(tl.int32), 0)
        room = k - above
        eq_seen = 0
        for s in range(0, L, BLOCK):
            o = s + tl.arange(0, BLOCK)
            key = _okey(tl.load(row + o, mask=o < L, other=-1))
            gt = (o < L) & (key > thr)
            eq = (o < L) & (key == thr)
            eq_rank = eq_seen + tl.cumsum(eq.to(tl.int32), 0) - eq.to(tl.int32)
            take = gt | (eq & (eq_rank < room))
            ti = take.to(tl.int32)
            pos = base + tl.cumsum(ti, 0) - ti
            if TRANSFORM:
                slot = (
                    tl.load(pt + (o >> PAGE_SHIFT), mask=take, other=0) << PAGE_SHIFT
                ) + (o & ((1 << PAGE_SHIFT) - 1))
            else:
                slot = o
            tl.store(out_ptr + r * stride_o + pos, slot, mask=take)
            base += tl.sum(ti, 0)
            eq_seen += tl.sum(eq.to(tl.int32), 0)
        tl.store(
            out_ptr + r * stride_o + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=out_cols >= base,
        )
    elif n_all >= k:
        n = n_all
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
        # one packed CAND-wide scan carries both ranks (each field < CAND <= 65536)
        cs = tl.cumsum(gti + (eqi * 65536), 0)
        rank_gt = (cs & 65535) - gti
        rank_eq = (cs // 65536) - eqi
        above = tl.sum(gti, 0)
        room = k - above
        take = gt | (eq & (rank_eq < room))
        n_take = tl.sum(take.to(tl.int32), 0)
        pos = rank_gt + tl.minimum(rank_eq, room)
        idx = tl.load(cidx_ptr + cb + o, mask=take, other=0)
        if TRANSFORM:
            slot = (
                tl.load(pt + (idx >> PAGE_SHIFT), mask=take, other=0) << PAGE_SHIFT
            ) + (idx & ((1 << PAGE_SHIFT) - 1))
        else:
            slot = idx
        tl.store(
            out_ptr + r * stride_o + out_cols,
            tl.full((TOPK,), -1, tl.int32),
            mask=out_cols >= n_take,
        )
        tl.store(out_ptr + r * stride_o + pos, slot, mask=take)


_SCRATCH: dict = {}


def _scratch(device: torch.device):
    """Fixed-address scratch (CUDA-graph safe), sized for MAX_ROWS rows."""
    key = (device.type, device.index)
    buf = _SCRATCH.get(key)
    if buf is None:
        buf = (
            torch.empty((MAX_ROWS, _CSTR), dtype=torch.int32, device=device),
            torch.empty((MAX_ROWS + 1,), dtype=torch.int32, device=device),
            torch.empty((MAX_ROWS, _CAND), dtype=torch.int32, device=device),
            torch.empty((MAX_ROWS, _CAND), dtype=torch.int32, device=device),
        )
        _SCRATCH[key] = buf
    return buf


def supported(logits: torch.Tensor, topk: int) -> bool:
    rows = logits.shape[0]
    return (
        0 < rows <= MAX_ROWS
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and rows * logits.stride(0) * 4 < 2**31
        and 0 < topk <= 2048
        and topk & (topk - 1) == 0
    )


def triton_top_k_per_row_decode(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    out: torch.Tensor,
    topk: int,
    page_table: torch.Tensor | None = None,
    page_size: int = 1,
) -> torch.Tensor:
    """Top-``topk`` columns of ``logits[row, :lengths[row]]`` into ``out[row, :topk]``
    (int32, ``-1`` padded, order undefined). Without ``page_table`` the indices are
    token positions inside the row; with it, physical KV slots
    ``page_table[row, p // page_size] * page_size + p % page_size``.

    ``logits`` fp32 ``[rows, >= max(lengths)]`` with unit column stride, ``lengths``
    int32 ``[rows]``; ``rows <= MAX_ROWS`` and ``rows * logits.stride(0) * 4 < 2**31``.
    """
    rows = logits.shape[0]
    assert supported(logits, topk), (tuple(logits.shape), logits.stride(), topk)
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[0] >= rows
    assert lengths.dtype == torch.int32 and lengths.numel() >= rows
    transform = page_table is not None
    if transform:
        assert page_size & (page_size - 1) == 0, page_size
    ctr, pref, ckey, cidx = _scratch(logits.device)
    li = logits.view(torch.int32)
    sl = li.stride(0)
    _topk_est[(rows,)](
        li,
        lengths,
        pref,
        ctr,
        sl,
        rows,
        CH=_SMP_CH,
        NCH=_SMP_NCH,
        TOPK=topk,
        OVER=_OVER,
        T=_T,
        RMAX=MAX_ROWS,
        CSTR=_CSTR,
        TIE_MIN=_TIE_MIN,
        num_warps=8,
    )
    _topk_collect[(_GRID,)](
        li,
        lengths,
        pref,
        ctr,
        ckey,
        cidx,
        rows,
        sl,
        T=_T,
        CAND=_CAND,
        RMAX=MAX_ROWS,
        GRID=_GRID,
        CSTR=_CSTR,
        NW=_COLLECT_NW,
        num_warps=_COLLECT_NW,
    )
    _topk_select[(rows,)](
        ckey,
        cidx,
        ctr,
        lengths,
        li,
        sl,
        page_table if transform else lengths,
        page_table.stride(0) if transform else 0,
        out,
        out.stride(0),
        CAND=_CAND,
        TOPK=topk,
        CSTR=_CSTR,
        PAGE_SHIFT=page_size.bit_length() - 1,
        BLOCK=_FALLBACK_BLOCK,
        TRANSFORM=transform,
        num_warps=16,  # CAND-wide (16K) register arrays
    )
    return out
