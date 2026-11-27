# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exact top-k (k <= 2048, a power of two) for the DSA indexer PREFILL rows (ROCm): many rows (up to the chunk's
query tokens, 4096 at max_num_batched_tokens 4096) of long rows (up to ~1M), fp32 logits, per-row [start, end).

Why: the prefill runs aiter's one-block-per-row radix top-k (`radix_topk_one_block_{decode,prefill}_mixed_k2048`):
one 1024-thread block per row reads its row from HBM two to three times (14-bit histogram pass, bucket compaction pass,
emit re-scan when the bucket does not fit LDS). A prefill call is sized by VLLM_SPARSE_INDEXER_MAX_LOGITS_MB (512 MB:
rows x width <= 128M fp32), so it is ~2-3 x 512 MB of reads per call (a54B ptrace: ~250 us per full call).

How (the decode port's est -> collect -> select, reorganised for many rows so the row is read ONCE):
  est      per row, sample <= 8192 logits (probes of 4, ~one probe per 256 elements, so the sample costs ~1/8 of
           the row's 128-byte lines at most) and take the exact t-th largest sample key by four 8-bit radix passes
           (t ~ OVER * k * samples / L); threshold = that key rounded down to 2**15 (only adds candidates); the tie
           guard of the decode port (`strict`: if the t-th sample key fills a large part of the row, collect only
           keys above it and fill the rest with tied elements).  Fused into collect when a row is one segment.
  collect grid (rows, SPLIT): program (r, s) scans its contiguous segment of row r IN ORDER and appends every key
           >= threshold, with its position, to its OWN candidate slot (no atomics), so each segment's candidates are
           in position order.  SPLIT = ceil(PROGRAMS / rows) (1 from 1024 rows on), so a call always has ~1024
           streaming programs: few long rows are split, many rows are not.
  select   per row: exact k-th largest candidate key (four 8-bit radix passes over the row's candidates), then the
           taken candidates are written segment by segment, in position order -> the output is ascending (-1 padded),
           like aiter's scan-order output, with no sort.
Exactness: if a segment overflows its slot (CAND) or the row has fewer than k candidates, select falls back to an
exact whole-row selection for that row (one program, slow, rare).  Tie mode: all candidates (> tie key) plus the first
tied elements in row order (exact: tied elements are interchangeable); those rows are then sorted by a last kernel.
Rows with L <= k take [0, L) and -1.  Indices are positions relative to the row start (what vLLM's prefill top-k
returns after `_localize_aiter_prefill_topk`).

Scratch (allocated once per device): 2 x SLOTS x CAND int32 = 256 MB at the defaults; rows beyond SLOTS / SPLIT are
processed in groups (2 groups at 4096 rows).  Launch shapes depend on the row count only (no host sync).
Knobs (env, read at import; the test also sets the module globals): VPD_TOPK_PREFILL_{CAND,SLOTS,PROGRAMS,BLOCK,NW}.
"""

from __future__ import annotations

import os
import sys

import torch
import triton
import triton.language as tl

# _okey: order-preserving int32 key of a float32 bit pattern; _row_threshold: exact k-th largest key of a whole row
# (4 radix passes).  Shared with the decode port so both paths have one definition.
from vpd_triton_topk_decode import _okey, _row_threshold


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


_CAND = _env_int("VPD_TOPK_PREFILL_CAND", 16384)  # candidates per segment slot
_OVER = 4  # candidate target, x topk (window [k, CAND] = [0.25, 2] x target at k = 2048)
_CH = 4  # probe length
_NCH = 2048  # max probes per row (sample vector = 8192)
_SPACING = 256  # one probe per this many elements (rows <= 512K) ...
_MIN_PROBES = 256  # ... but at least this many probes; rows <= 8192 are sampled whole
_TIE_MIN = _CAND // 4
_SLOTS = _env_int("VPD_TOPK_PREFILL_SLOTS", 2048)  # candidate slots (programs alive between collect and select)
_PROGRAMS = _env_int("VPD_TOPK_PREFILL_PROGRAMS", 1024)  # collect programs a call aims for
_MAX_SPLIT = 16
_BLOCK = _env_int("VPD_TOPK_PREFILL_BLOCK", 4096)  # collect / fallback scan block (elements)
_COLLECT_NW = _env_int("VPD_TOPK_PREFILL_NW", 8)
_EST_NW = 8
_CB = 2048  # select block over candidates
_SELECT_NW = 4
_CSTR = 8  # int32 per row in ctr: 0 collect threshold, 2 tie key, 3 strict
_SORT = os.environ.get("VPD_TOPK_PREFILL_SORT", "1") != "0"  # sort tie-mode rows (the only unsorted ones)

MODE_NAMES = ("normal", "tie", "whole_row", "short")


@triton.jit
def _kth_vec(key, valid, t):
    """Exact t-th largest of key[valid] (int32 order keys, t >= 1): four 8-bit radix passes on u = key ^ INT_MIN,
    whose unsigned order is the key order (same scheme as _row_threshold, on a register vector)."""
    bins = tl.arange(0, 512)
    zero = t * 0
    ff = zero + 255
    prefix = zero
    pmask = zero
    need = t
    u = key ^ (-2147483648)
    for p in tl.static_range(0, 4):
        hit = valid & ((u & pmask) == prefix)
        d = tl.where(hit, (u >> (24 - 8 * p)) & ff, 511)
        h = tl.histogram(d, 512)
        h = tl.where(bins < 256, h, 0)
        ge = tl.sum(h, 0) - tl.cumsum(h, 0) + h  # elements in bins >= b
        b = tl.max(tl.where(ge >= need, bins, -1), 0)
        need -= tl.sum(tl.where(bins > b, h, 0), 0)
        prefix = prefix | (b.to(tl.int32) << (24 - 8 * p))
        pmask = pmask | (ff << (24 - 8 * p))
    return prefix ^ (-2147483648)


@triton.jit
def _est_row(row, L, k, CH: tl.constexpr, NCH: tl.constexpr, SPACING: tl.constexpr, MIN_PROBES: tl.constexpr,
             OVER: tl.constexpr, TIE_MIN: tl.constexpr):
    """Per-row collect threshold (row = int32 view of the row's first element, L > k)."""
    o = tl.arange(0, CH * NCH)
    nprobe = tl.where(L <= CH * NCH, NCH, tl.minimum(NCH, tl.maximum(MIN_PROBES, L // SPACING)))
    span = tl.maximum(CH, L // nprobe)
    p = (o // CH) * span + (o % CH)
    m = ((o // CH) < nprobe) & (p < L)
    key = _okey(tl.load(row + p, mask=m, other=-1))
    ns = tl.sum(m.to(tl.int32), 0)
    t = tl.maximum(1, tl.minimum((OVER * k * ns) // tl.maximum(L, 1), ns))
    kt = _kth_vec(key, m, t)
    thr = kt & (-32768)  # rounded down: only adds candidates
    n_eq = tl.sum((m & (key == kt)).to(tl.int32), 0)
    strict = ((n_eq.to(tl.int64) * L) // tl.maximum(ns, 1) > TIE_MIN) & (kt < 2147483647)
    cthr = tl.where(strict, kt + 1, thr)
    return cthr, kt, strict.to(tl.int32)


@triton.jit
def _row_bounds(starts_ptr, ends_ptr, r, HAS_START: tl.constexpr):
    en = tl.load(ends_ptr + r)
    if HAS_START:
        st = tl.load(starts_ptr + r)
    else:
        st = en * 0
    return st, tl.maximum(en - st, 0)


@triton.jit
def _pf_est(li_ptr, starts_ptr, ends_ptr, ctr_ptr, stride_l, row0,
            HAS_START: tl.constexpr, TOPK: tl.constexpr, CSTR: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr,
            SPACING: tl.constexpr, MIN_PROBES: tl.constexpr, OVER: tl.constexpr, TIE_MIN: tl.constexpr):
    rr = tl.program_id(0)
    r = row0 + rr
    st, L = _row_bounds(starts_ptr, ends_ptr, r, HAS_START)
    if L > TOPK:
        row = li_ptr + r.to(tl.int64) * stride_l + st
        cthr, kt, strict = _est_row(row, L, TOPK, CH, NCH, SPACING, MIN_PROBES, OVER, TIE_MIN)
        tl.store(ctr_ptr + rr * CSTR, cthr)
        tl.store(ctr_ptr + rr * CSTR + 2, kt)
        tl.store(ctr_ptr + rr * CSTR + 3, strict)


@triton.jit
def _pf_collect(li_ptr, starts_ptr, ends_ptr, ctr_ptr, cnt_ptr, ckey_ptr, cidx_ptr, stride_l, row0, split,
                HAS_START: tl.constexpr, FUSE_EST: tl.constexpr, TOPK: tl.constexpr, CAND: tl.constexpr,
                BLOCK: tl.constexpr, CSTR: tl.constexpr, CH: tl.constexpr, NCH: tl.constexpr,
                SPACING: tl.constexpr, MIN_PROBES: tl.constexpr, OVER: tl.constexpr, TIE_MIN: tl.constexpr):
    rr = tl.program_id(0)
    s = tl.program_id(1)
    r = row0 + rr
    st, L = _row_bounds(starts_ptr, ends_ptr, r, HAS_START)
    slot = rr * split + s
    n = L * 0
    if L > TOPK:
        row = li_ptr + r.to(tl.int64) * stride_l + st
        if FUSE_EST:
            cthr, kt, strict = _est_row(row, L, TOPK, CH, NCH, SPACING, MIN_PROBES, OVER, TIE_MIN)
            tl.store(ctr_ptr + rr * CSTR, cthr)
            tl.store(ctr_ptr + rr * CSTR + 2, kt)
            tl.store(ctr_ptr + rr * CSTR + 3, strict)
        else:
            cthr = tl.load(ctr_ptr + rr * CSTR)
        seg = (((L + split - 1) // split) + BLOCK - 1) // BLOCK * BLOCK
        lo = s * seg
        hi = tl.minimum(lo + seg, L)
        cb = slot.to(tl.int64) * CAND
        for b0 in range(lo, hi, BLOCK):
            o = b0 + tl.arange(0, BLOCK)
            mm = o < hi
            key = _okey(tl.load(row + o, mask=mm, other=-1))
            sel = mm & (key >= cthr)
            si = sel.to(tl.int32)
            c = tl.sum(si, 0)
            if c > 0:
                pos = n + tl.cumsum(si, 0) - si
                w = sel & (pos < CAND)
                tl.store(ckey_ptr + cb + pos, key, mask=w)
                tl.store(cidx_ptr + cb + pos, o, mask=w)
            n += c
    tl.store(cnt_ptr + slot, n)


@triton.jit
def _pf_select(li_ptr, starts_ptr, ends_ptr, ctr_ptr, cnt_ptr, ckey_ptr, cidx_ptr, out_ptr, mode_ptr,
               stride_l, stride_o, row0, split,
               HAS_START: tl.constexpr, TOPK: tl.constexpr, CAND: tl.constexpr, CSTR: tl.constexpr,
               CB: tl.constexpr, BLOCK: tl.constexpr):
    rr = tl.program_id(0)
    r = row0 + rr
    st, L = _row_bounds(starts_ptr, ends_ptr, r, HAS_START)
    outr = out_ptr + r.to(tl.int64) * stride_o
    cols = tl.arange(0, TOPK)
    zero = L * 0
    mode = zero + 3
    if L <= TOPK:
        tl.store(outr + cols, tl.where(cols < L, cols, -1))
    else:
        row = li_ptr + r.to(tl.int64) * stride_l + st
        kt = tl.load(ctr_ptr + rr * CSTR + 2)
        strict = tl.load(ctr_ptr + rr * CSTR + 3)
        N = zero
        ovf = zero
        for s in range(0, split):
            c = tl.load(cnt_ptr + rr * split + s)
            N += c
            ovf = ovf | (c > CAND).to(tl.int32)
        mode = tl.where((ovf != 0) | ((strict == 0) & (N < TOPK)), 2, tl.where(N < TOPK, 1, 0))
        if mode == 0:
            # exact k-th largest candidate key
            bins = tl.arange(0, 512)
            ff = zero + 255
            prefix = zero
            pmask = zero
            need = zero + TOPK
            for p in tl.static_range(0, 4):
                h = tl.zeros((512,), tl.int32)
                for s in range(0, split):
                    c = tl.load(cnt_ptr + rr * split + s)
                    cb = (rr * split + s).to(tl.int64) * CAND
                    for b0 in range(0, c, CB):
                        o = b0 + tl.arange(0, CB)
                        mm = o < c
                        u = tl.load(ckey_ptr + cb + o, mask=mm, other=0) ^ (-2147483648)
                        hit = mm & ((u & pmask) == prefix)
                        d = tl.where(hit, (u >> (24 - 8 * p)) & ff, 511)
                        h += tl.histogram(d, 512)
                h = tl.where(bins < 256, h, 0)
                ge = tl.sum(h, 0) - tl.cumsum(h, 0) + h
                b = tl.max(tl.where(ge >= need, bins, -1), 0)
                need -= tl.sum(tl.where(bins > b, h, 0), 0)
                prefix = prefix | (b.to(tl.int32) << (24 - 8 * p))
                pmask = pmask | (ff << (24 - 8 * p))
            thr = prefix ^ (-2147483648)
            room = need  # elements equal to thr to take (the first ones in row order)
            base = zero
            eq_seen = zero
            for s in range(0, split):
                c = tl.load(cnt_ptr + rr * split + s)
                cb = (rr * split + s).to(tl.int64) * CAND
                for b0 in range(0, c, CB):
                    o = b0 + tl.arange(0, CB)
                    mm = o < c
                    key = tl.load(ckey_ptr + cb + o, mask=mm, other=-2147483648)
                    gti = (mm & (key > thr)).to(tl.int32)
                    eqi = (mm & (key == thr)).to(tl.int32)
                    # one packed scan carries both ranks (each < CB <= 65536)
                    cs = tl.cumsum(gti + eqi * 65536, 0)
                    eq_rank = eq_seen + (cs // 65536) - eqi
                    take = (gti != 0) | ((eqi != 0) & (eq_rank < room))
                    ti = take.to(tl.int32)
                    pos = base + tl.cumsum(ti, 0) - ti
                    idx = tl.load(cidx_ptr + cb + o, mask=take, other=0)
                    tl.store(outr + pos, idx, mask=take)
                    base += tl.sum(ti, 0)
                    eq_seen += tl.sum(eqi, 0)
        if mode == 1:
            # Tie mode with fewer candidates than k: every candidate (> kt), then tied elements in row order.
            base = zero
            for s in range(0, split):
                c = tl.load(cnt_ptr + rr * split + s)
                cb = (rr * split + s).to(tl.int64) * CAND
                for b0 in range(0, c, CB):
                    o = b0 + tl.arange(0, CB)
                    mm = o < c
                    idx = tl.load(cidx_ptr + cb + o, mask=mm, other=0)
                    tl.store(outr + base + o, idx, mask=mm)
                base += c
            need = TOPK - N
            filled = zero
            s0 = zero
            while (filled < need) & (s0 < L):
                o = s0 + tl.arange(0, BLOCK)
                key = _okey(tl.load(row + o, mask=o < L, other=-1))
                eqi = ((o < L) & (key == kt)).to(tl.int32)
                rank = filled + tl.cumsum(eqi, 0) - eqi
                take = (eqi != 0) & (rank < need)
                tl.store(outr + N + rank, o, mask=take)
                filled += tl.sum(take.to(tl.int32), 0)
                s0 += BLOCK
            mode = tl.where(filled == need, 1, 2)
        if mode == 2:
            # Exact whole-row selection (overflow / underflow): rare and slow; writes the row in position order.
            thr = _row_threshold(row, L, zero + TOPK, BLOCK)
            above = zero
            for q0 in range(0, L, BLOCK):
                o = q0 + tl.arange(0, BLOCK)
                key = _okey(tl.load(row + o, mask=o < L, other=-1))
                above += tl.sum(((o < L) & (key > thr)).to(tl.int32), 0)
            room = TOPK - above
            base = zero
            eq_seen = zero
            for q0 in range(0, L, BLOCK):
                o = q0 + tl.arange(0, BLOCK)
                key = _okey(tl.load(row + o, mask=o < L, other=-1))
                gt = (o < L) & (key > thr)
                eqi = ((o < L) & (key == thr)).to(tl.int32)
                eq_rank = eq_seen + tl.cumsum(eqi, 0) - eqi
                take = gt | ((eqi != 0) & (eq_rank < room))
                ti = take.to(tl.int32)
                pos = base + tl.cumsum(ti, 0) - ti
                tl.store(outr + pos, o, mask=take)
                base += tl.sum(ti, 0)
                eq_seen += tl.sum(eqi, 0)
    tl.store(mode_ptr + r, mode)


@triton.jit
def _pf_sort(out_ptr, stride_o, mode_ptr, TOPK: tl.constexpr):
    """Sort a tie-mode row ascending (the only rows select does not write in position order; all TOPK valid)."""
    r = tl.program_id(0)
    if tl.load(mode_ptr + r) == 1:
        c = tl.arange(0, TOPK)
        p = out_ptr + r.to(tl.int64) * stride_o + c
        tl.store(p, tl.sort(tl.load(p)))


_SCR: dict = {}


def _scratch(device: torch.device, rows: int):
    key = (device.type, device.index)
    s = _SCR.get(key)
    if s is None or s["cand"] != _CAND or s["slots"] != _SLOTS:
        s = dict(
            cand=_CAND,
            slots=_SLOTS,
            ctr=torch.zeros((_SLOTS, _CSTR), dtype=torch.int32, device=device),
            cnt=torch.zeros((_SLOTS,), dtype=torch.int32, device=device),
            ckey=torch.empty((_SLOTS, _CAND), dtype=torch.int32, device=device),
            cidx=torch.empty((_SLOTS, _CAND), dtype=torch.int32, device=device),
            mode=torch.zeros((4096,), dtype=torch.int32, device=device),
        )
        _SCR[key] = s
    if s["mode"].numel() < rows:
        s["mode"] = torch.zeros((max(rows, 2 * s["mode"].numel()),), dtype=torch.int32, device=device)
    return s


def supported(logits: torch.Tensor, topk: int) -> bool:
    return (
        logits.dim() == 2
        and logits.shape[0] > 0
        and logits.dtype == torch.float32
        and logits.stride(1) == 1
        and 0 < topk <= 2048
        and topk & (topk - 1) == 0
    )


def plan(rows: int) -> tuple[int, int]:
    """(SPLIT, rows per group) of a call with `rows` rows."""
    split = max(1, min(_MAX_SPLIT, -(-_PROGRAMS // rows)))
    return split, max(1, min(rows, _SLOTS // split))


def triton_top_k_per_row_prefill(
    logits: torch.Tensor,
    row_starts: torch.Tensor | None,
    row_ends: torch.Tensor,
    out: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Top-``topk`` of ``logits[r, row_starts[r]:row_ends[r]]`` (``row_starts`` None: 0) into ``out[r, :topk]``:
    int32 positions relative to the row start, ascending, -1 padded (rows shorter than topk).  Exact: the selected
    values are the row's top-k values; among elements tied at the k-th value the choice is unspecified.
    ``logits`` fp32 [rows, W] with unit column stride, bounds int32 [rows] with 0 <= start <= end <= W."""
    rows = logits.shape[0]
    assert supported(logits, topk), (tuple(logits.shape), logits.stride(), logits.dtype, topk)
    assert out.dtype == torch.int32 and out.stride(1) == 1 and out.shape[0] >= rows and out.shape[1] >= topk
    assert row_ends.dtype == torch.int32 and row_ends.numel() >= rows and row_ends.is_contiguous()
    has_start = row_starts is not None
    if has_start:
        assert row_starts.dtype == torch.int32 and row_starts.numel() >= rows and row_starts.is_contiguous()
    s = _scratch(logits.device, rows)
    li = logits.view(torch.int32)
    sl = li.stride(0)
    starts = row_starts if has_start else row_ends
    split, group = plan(rows)
    est = dict(CH=_CH, NCH=_NCH, SPACING=_SPACING, MIN_PROBES=_MIN_PROBES, OVER=_OVER, TIE_MIN=_TIE_MIN)
    for row0 in range(0, rows, group):
        g = min(group, rows - row0)
        if split > 1:
            _pf_est[(g,)](li, starts, row_ends, s["ctr"], sl, row0, HAS_START=has_start, TOPK=topk, CSTR=_CSTR,
                          num_warps=_EST_NW, **est)
        _pf_collect[(g, split)](
            li, starts, row_ends, s["ctr"], s["cnt"], s["ckey"], s["cidx"], sl, row0, split,
            HAS_START=has_start, FUSE_EST=split == 1, TOPK=topk, CAND=_CAND, BLOCK=_BLOCK, CSTR=_CSTR,
            num_warps=_COLLECT_NW, **est,
        )
        _pf_select[(g,)](
            li, starts, row_ends, s["ctr"], s["cnt"], s["ckey"], s["cidx"], out, s["mode"], sl, out.stride(0),
            row0, split, HAS_START=has_start, TOPK=topk, CAND=_CAND, CSTR=_CSTR, CB=_CB, BLOCK=_BLOCK,
            num_warps=_SELECT_NW,
        )
    if _SORT:
        _pf_sort[(rows,)](out, out.stride(0), s["mode"], TOPK=topk, num_warps=8)
    return out


def last_modes(device: torch.device, rows: int) -> torch.Tensor:
    """Per-row path of the last call on `device` (0 normal, 1 tie, 2 whole-row fallback, 3 short), for tests / checks."""
    return _SCR[(device.type, device.index)]["mode"][:rows].clone()


def check_call(logits, row_starts, row_ends, out, topk) -> dict:
    """Compare one call's output with torch.topk (host sync; for tests and the hook's VPD_TOPK_PREFILL_CHECK).
    bad = rows whose selection is not an exact top-k (count, range, duplicates, values) or, with sorting on,
    not ascending; also the per-path row counts of that call."""
    rows, W = logits.shape
    dev = logits.device
    st = row_starts[:rows].long() if row_starts is not None else torch.zeros(rows, dtype=torch.long, device=dev)
    L = (row_ends[:rows].long() - st).clamp(min=0)
    kk = L.clamp(max=topk)
    col = torch.arange(W, device=dev, dtype=torch.int32)[None, :]
    st32, en32 = st.int()[:, None], (st + L).int()[:, None]
    masked = torch.where((col >= st32) & (col < en32), logits, float("-inf"))
    del col
    kref = min(topk, W)
    ref = torch.topk(masked, kref, dim=1).values
    if kref < topk:
        ref = torch.cat([ref, torch.full((rows, topk - kref), float("-inf"), device=dev)], 1)
    del masked
    sel = out[:rows, :topk].long()
    valid = sel >= 0
    cnt = valid.sum(1)
    col = torch.arange(topk, device=dev)[None, :]
    in_k = col < kk[:, None]
    oor = (valid & (sel >= L[:, None])).any(1)
    g = torch.gather(logits, 1, (sel.clamp(min=0) + st[:, None]).clamp(max=W - 1))
    vals = torch.where(valid, g, float("-inf")).sort(1, descending=True).values
    val_bad = ((vals != ref) & in_k).any(1)
    si = torch.where(valid, sel, 2**40).sort(1).values
    dup = ((si[:, 1:] == si[:, :-1]) & (si[:, 1:] != 2**40)).any(1)
    pad_bad = (valid != in_k).any(1)  # valid entries first, exactly kk of them
    order_bad = ((sel[:, 1:] <= sel[:, :-1]) & valid[:, 1:]).any(1) if _SORT else torch.zeros_like(dup)
    bad = (cnt != kk) | oor | dup | val_bad | pad_bad | order_bad
    modes = last_modes(dev, rows)
    res = dict(rows=rows, bad=int(bad.sum()), cnt_bad=int((cnt != kk).sum()), oor=int(oor.sum()),
               dup=int(dup.sum()), val_bad=int(val_bad.sum()), pad_bad=int(pad_bad.sum()),
               order_bad=int(order_bad.sum()))
    for i, n in enumerate(MODE_NAMES):
        res[n] = int((modes == i).sum())
    if res["bad"]:
        r = int(bad.int().argmax())
        res["first_bad_row"] = r
        res["first_bad_L"] = int(L[r])
        res["first_bad_mode"] = int(modes[r])
    return res


_CHK = {"calls": 0, "checked": 0, "bad_rows": 0, "rows": 0, "modes": [0, 0, 0, 0], "logged": False}


def hook_call(logits, row_starts, row_ends, out, topk, where: str) -> None:
    """The overlay hooks' entry: run the kernel; log the first call; with VPD_TOPK_PREFILL_CHECK=N compare every
    N-th call with torch.topk and print running counters (host sync on those calls only)."""
    triton_top_k_per_row_prefill(logits, row_starts, row_ends, out, topk)
    _CHK["calls"] += 1
    if not _CHK["logged"]:
        _CHK["logged"] = True
        print(f"[vpd-topk-prefill] pid {os.getpid()} first call ({where}): rows {logits.shape[0]} width "
              f"{logits.shape[1]} split/group {plan(logits.shape[0])}", file=sys.stderr, flush=True)
    every = _env_int("VPD_TOPK_PREFILL_CHECK", 0)
    if every > 0 and _CHK["calls"] % every == 0:
        res = check_call(logits, row_starts, row_ends, out, topk)
        _CHK["checked"] += 1
        _CHK["bad_rows"] += res["bad"]
        _CHK["rows"] += res["rows"]
        for i, n in enumerate(MODE_NAMES):
            _CHK["modes"][i] += res[n]
        print(f"[vpd-topk-prefill] pid {os.getpid()} {where}: calls {_CHK['calls']} checked {_CHK['checked']} rows "
              f"{_CHK['rows']} bad_rows {_CHK['bad_rows']} modes normal/tie/whole_row/short "
              f"{'/'.join(map(str, _CHK['modes']))} last {res}", file=sys.stderr, flush=True)
