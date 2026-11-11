#!/usr/bin/env python3
"""Build kf_sparse_mla_split2g.py: the production KF sparse MLA with the split-K decode kernel reading a > 2 GiB KV pool
through two 32-bit-addressable halves.

Triton 3.7's HIP backend marks a pointer argument as 32-bit range (tt.pointer_range = 32, which buffer loads need) only
when the whole storage behind it is <= 2^31 - 1 bytes, so a > 2 GiB pool loses buffer loads for every KV gather, and
the kernel also switches to 64-bit row offsets. Here the launcher passes the pool as two pointer objects whose
ptr_range() is each half's size (rows [0, S) and [S, N)), and the kernel issues each gather from both halves with
complementary masks; a masked-off buffer load returns 0 without a memory access, so OR-ing the two is exact.

Usage: python3 make_kf_sparse_split2g.py SRC DST [other]   (other: the upper-half load takes the lower half's value as
       its masked-off default instead of OR-ing two zero-filled loads)
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
OTHER_LO_DEFAULT = len(sys.argv) > 3 and sys.argv[3] == "other"
s = open(src).read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"anchor found {s.count(old)} times, want {count}: {old[:80]!r}"
    s = s.replace(old, new)


sub(
    "def _page_offsets_fit_i32(num_pages: int, kv_dim: int) -> bool:\n",
    '''class _KvRows:
    """Rows [start, start + n) of a contiguous KV pool as a Triton pointer argument. Triton's HIP backend specializes
    a pointer as 32-bit range (buffer loads) from the size of the tensor's whole storage, so a view into a > 2 GiB pool
    never qualifies; ptr_range() gives the size of these rows instead."""

    def __init__(self, kv: torch.Tensor, start: int, n: int):
        row = kv.stride(0) * kv.element_size()
        self.dtype = kv.dtype
        self._ptr = kv.data_ptr() + start * row
        self._range = n * row

    def data_ptr(self) -> int:
        return self._ptr

    def ptr_range(self) -> int:
        return self._range


def _kv_split_rows(kv: torch.Tensor) -> int:
    """First row of the upper half when a > 2 GiB pool can be read as two halves of <= 2^31 - 1 bytes, else 0."""
    row = kv.stride(0) * kv.element_size()
    if not kv.is_contiguous() or kv.shape[0] * row <= _I32_MAX:
        return 0
    lo = _I32_MAX // row
    return lo if (kv.shape[0] - lo) * row <= _I32_MAX else 0


@triton.jit
def _gather_kv2(base_lo, base_hi, offs, m_lo, m_hi, CACHE_MOD: tl.constexpr, KV_BITS: tl.constexpr,
                OTHER_LO: tl.constexpr):
    """One KV gather from both halves; exactly one mask is set per row, the other load returns 0 bits."""
    lo = tl.load(base_lo + offs, mask=m_lo, other=0.0, cache_modifier=CACHE_MOD)
    if OTHER_LO:
        # the row mask is uniform across each vector, so this is one select per dword, not per byte like `|`
        return tl.load(base_hi + offs, mask=m_hi, other=lo, cache_modifier=CACHE_MOD)
    hi = tl.load(base_hi + offs, mask=m_hi, other=0.0, cache_modifier=CACHE_MOD)
    if KV_BITS == 8:
        bits = lo.to(tl.uint8, bitcast=True) | hi.to(tl.uint8, bitcast=True)
    else:
        bits = lo.to(tl.uint16, bitcast=True) | hi.to(tl.uint16, bitcast=True)
    return bits.to(lo.dtype, bitcast=True)


def _page_offsets_fit_i32(num_pages: int, kv_dim: int) -> bool:
''',
)

# split-K kernel signature: the upper half's pointer and first row, and the switch
sub(
    """    lse_partial_ptr,
    acc_partial_ptr,
    qk_scale,
    fp8_max,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
    KV_SPLITS: tl.constexpr,
""",
    """    lse_partial_ptr,
    acc_partial_ptr,
    qk_scale,
    fp8_max,
    kv_hi_ptr,
    kv_hi_start,
    topk: tl.constexpr,
    H: tl.constexpr,
    KV_DIM: tl.constexpr,
    D_V: tl.constexpr,
    D_TAIL: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    STRIDE_QN_T: tl.constexpr,
    STRIDE_QN_H: tl.constexpr,
    STRIDE_QR_T: tl.constexpr,
    STRIDE_QR_H: tl.constexpr,
    USE_FP8_DOT: tl.constexpr,
    KV_SPLITS: tl.constexpr,
""",
)
sub(
    """    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
):
    \"\"\"Split-K partial kernel. grid=(seq, head_blocks, kv_splits).
""",
    """    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
    SPLIT_KV: tl.constexpr = False,
    KV_BITS: tl.constexpr = 8,
    OTHER_LO: tl.constexpr = False,
):
    \"\"\"Split-K partial kernel. grid=(seq, head_blocks, kv_splits).

    ``SPLIT_KV``: the pool is read as rows [0, kv_hi_start) through ``kv_ptr`` and the rest through ``kv_hi_ptr``,
    each <= 2^31 - 1 bytes so both keep 32-bit buffer loads (see :class:`_KvRows`); every gather is issued from both
    with complementary masks.
""",
)

old_gather = """        page = tl.where(valid, slot, 0)
        if USE_I64_PAGE:
            kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
        else:
            kv_base = kv_ptr + page[:, None] * KV_DIM

        kv0 = tl.load(
            kv_base + g[None, :],
            mask=valid[:, None],
            other=0.0,
            cache_modifier=KV_CACHE_MOD,
        ).to(input_type)
        if NUM_GROUPS >= 2:
            kv1 = tl.load(
                kv_base + (GW + g)[None, :],
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
        if NUM_GROUPS >= 3:
            kv2 = tl.load(
                kv_base + (2 * GW + g)[None, :],
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
        if NUM_GROUPS >= 4:
            kv3 = tl.load(
                kv_base + (3 * GW + g)[None, :],
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
        kv_tail = tl.load(
            kv_base + (D_V + dt)[None, :],
            mask=valid[:, None],
            other=0.0,
            cache_modifier=KV_CACHE_MOD,
        ).to(input_type)
"""
new_gather = """        page = tl.where(valid, slot, 0)
        if SPLIT_KV:
            in_hi = page >= kv_hi_start
            m_lo = (valid & (page < kv_hi_start))[:, None]
            m_hi = (valid & in_hi)[:, None]
            base_lo = kv_ptr + tl.where(in_hi, 0, page)[:, None] * KV_DIM
            base_hi = kv_hi_ptr + tl.where(in_hi, page - kv_hi_start, 0)[:, None] * KV_DIM
            kv0 = _gather_kv2(base_lo, base_hi, g[None, :], m_lo, m_hi, KV_CACHE_MOD, KV_BITS, OTHER_LO).to(input_type)
            if NUM_GROUPS >= 2:
                kv1 = _gather_kv2(
                    base_lo, base_hi, (GW + g)[None, :], m_lo, m_hi, KV_CACHE_MOD, KV_BITS, OTHER_LO
                ).to(input_type)
            if NUM_GROUPS >= 3:
                kv2 = _gather_kv2(
                    base_lo, base_hi, (2 * GW + g)[None, :], m_lo, m_hi, KV_CACHE_MOD, KV_BITS, OTHER_LO
                ).to(input_type)
            if NUM_GROUPS >= 4:
                kv3 = _gather_kv2(
                    base_lo, base_hi, (3 * GW + g)[None, :], m_lo, m_hi, KV_CACHE_MOD, KV_BITS, OTHER_LO
                ).to(input_type)
            kv_tail = _gather_kv2(
                base_lo, base_hi, (D_V + dt)[None, :], m_lo, m_hi, KV_CACHE_MOD, KV_BITS, OTHER_LO
            ).to(input_type)
        else:
            if USE_I64_PAGE:
                kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
            else:
                kv_base = kv_ptr + page[:, None] * KV_DIM

            kv0 = tl.load(
                kv_base + g[None, :],
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
            if NUM_GROUPS >= 2:
                kv1 = tl.load(
                    kv_base + (GW + g)[None, :],
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MOD,
                ).to(input_type)
            if NUM_GROUPS >= 3:
                kv2 = tl.load(
                    kv_base + (2 * GW + g)[None, :],
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MOD,
                ).to(input_type)
            if NUM_GROUPS >= 4:
                kv3 = tl.load(
                    kv_base + (3 * GW + g)[None, :],
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MOD,
                ).to(input_type)
            kv_tail = tl.load(
                kv_base + (D_V + dt)[None, :],
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
"""
# the split-K kernel's gather is the only one written with this exact text
sub(old_gather, new_gather)

# launcher
sub(
    "    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)\n",
    "    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)\n"
    "    split_kv_hi = _kv_split_rows(kv) if decode_tuned and _SPLIT_2G else 0\n",
)
sub(
    """            lse_partial,
            acc_partial,
            qk_scale,
            _FP8_MAX,
            topk=topk,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=split_num_groups,
""",
    """            lse_partial,
            acc_partial,
            qk_scale,
            _FP8_MAX,
            _KvRows(kv, split_kv_hi, kv.shape[0] - split_kv_hi) if split_kv_hi else kv,
            split_kv_hi,
            topk=topk,
            H=H,
            KV_DIM=kv_dim,
            D_V=d_v,
            D_TAIL=d_tail,
            NUM_GROUPS=split_num_groups,
""",
)
sub(
    """        _sparse_mla_split_k_kernel[(seq, n_head_blocks, kv_splits)](
            q_nope,
            q_rope,
            kv,
            idx_flat,
""",
    """        _sparse_mla_split_k_kernel[(seq, n_head_blocks, kv_splits)](
            q_nope,
            q_rope,
            _KvRows(kv, 0, split_kv_hi) if split_kv_hi else kv,
            idx_flat,
""",
)
sub(
    """            USE_I64_PAGE=not split_i32_page,
            DENSE_TILES=split_dense_tiles,
""",
    """            USE_I64_PAGE=not (split_i32_page or split_kv_hi),
            DENSE_TILES=split_dense_tiles,
""",
)
sub(
    """            FIXED_TRIP=split_fixed_trip,
            num_warps=split_num_warps,
""",
    """            FIXED_TRIP=split_fixed_trip,
            SPLIT_KV=bool(split_kv_hi),
            KV_BITS=kv.element_size() * 8,
            OTHER_LO=_SPLIT_OTHER_LO,
            num_warps=split_num_warps,
""",
)
sub(
    "_I32_MAX = (1 << 31) - 1\n",
    "_I32_MAX = (1 << 31) - 1\n"
    "_SPLIT_2G = __import__(\"os\").environ.get(\"KF_SPARSE_SPLIT_2G\", \"1\") == \"1\"\n"
    f"_SPLIT_OTHER_LO = {OTHER_LO_DEFAULT}\n",
)
open(dst, "w").write(s)
print("wrote", dst)
