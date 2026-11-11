#!/usr/bin/env python3
"""Build kf_sparse_mla_u32off.py: the production KF sparse MLA whose split-K decode kernel addresses a 2-4 GiB KV pool
with one unsigned 32-bit offset per element (row * KV_DIM + column, formed before it touches the pointer), instead of
64-bit per-lane pointers. Triton zero-extends an unsigned offset added to a pointer, so every gather is the uniform
pool base plus a zero-extended 32-bit value, which the AMDGPU backend can issue as a global load with a scalar base and
a 32-bit vector offset. (The 11-09 unsigned attempt added the row to the pointer first and the column afterwards, which
leaves a 64-bit per-lane pointer.)

Usage: python3 make_kf_sparse_u32off.py SRC DST
"""
import sys

src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"anchor found {s.count(old)} times, want {count}: {old[:80]!r}"
    s = s.replace(old, new)


sub(
    """    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
):
    \"\"\"Split-K partial kernel. grid=(seq, head_blocks, kv_splits).
""",
    """    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
    U32_OFF: tl.constexpr = False,
):
    \"\"\"Split-K partial kernel. grid=(seq, head_blocks, kv_splits).

    ``U32_OFF``: each KV element is addressed as the pool base plus one unsigned 32-bit offset (pools up to 4 GiB),
    which the backend can issue as a scalar-base global load instead of a 64-bit per-lane pointer.
""",
)

old_gather = """        page = tl.where(valid, slot, 0)
        if USE_I64_PAGE:
            kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
        else:
            kv_base = kv_ptr + page[:, None] * KV_DIM

        kv0 = tl.load(
            kv_base + g[None, :],
"""
new_gather = """        page = tl.where(valid, slot, 0)
        if U32_OFF:
            row_u32 = page[:, None].to(tl.uint32) * KV_DIM
            kv0 = tl.load(
                kv_ptr + (row_u32 + g[None, :].to(tl.uint32)),
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
            if NUM_GROUPS >= 2:
                kv1 = tl.load(
                    kv_ptr + (row_u32 + (GW + g)[None, :].to(tl.uint32)),
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MOD,
                ).to(input_type)
            if NUM_GROUPS >= 3:
                kv2 = tl.load(
                    kv_ptr + (row_u32 + (2 * GW + g)[None, :].to(tl.uint32)),
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MOD,
                ).to(input_type)
            if NUM_GROUPS >= 4:
                kv3 = tl.load(
                    kv_ptr + (row_u32 + (3 * GW + g)[None, :].to(tl.uint32)),
                    mask=valid[:, None],
                    other=0.0,
                    cache_modifier=KV_CACHE_MOD,
                ).to(input_type)
            kv_tail = tl.load(
                kv_ptr + (row_u32 + (D_V + dt)[None, :].to(tl.uint32)),
                mask=valid[:, None],
                other=0.0,
                cache_modifier=KV_CACHE_MOD,
            ).to(input_type)
        elif USE_I64_PAGE:
            kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
        else:
            kv_base = kv_ptr + page[:, None] * KV_DIM

        if not U32_OFF:
            kv0 = tl.load(
            kv_base + g[None, :],
"""
sub(old_gather, new_gather)
# the rest of the original gather block moves under `if not U32_OFF:`
start = s.index("        if not U32_OFF:\n            kv0 = tl.load(\n")
end = s.index("        scores = tl.dot(q0, tl.trans(kv0))", start)
block = s[start:end]
head, rest = block.split("            kv0 = tl.load(\n", 1)
rest = "".join(("    " + ln if ln.strip() else ln) for ln in rest.splitlines(True))
s = s[:start] + head + "            kv0 = tl.load(\n" + rest + s[end:]

sub(
    "    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)\n",
    "    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)\n"
    "    split_u32_off = (\n"
    "        decode_tuned and _U32_OFF and not split_i32_page and kv.is_contiguous()\n"
    "        and kv.shape[0] * kv.stride(0) * kv.element_size() <= (1 << 32) - 1\n"
    "    )\n",
)
sub(
    """            USE_I64_PAGE=not split_i32_page,
            DENSE_TILES=split_dense_tiles,
""",
    """            USE_I64_PAGE=not (split_i32_page or split_u32_off),
            DENSE_TILES=split_dense_tiles,
""",
)
sub(
    """            FIXED_TRIP=split_fixed_trip,
            num_warps=split_num_warps,
""",
    """            FIXED_TRIP=split_fixed_trip,
            U32_OFF=split_u32_off,
            num_warps=split_num_warps,
""",
)
sub(
    "_I32_MAX = (1 << 31) - 1\n",
    "_I32_MAX = (1 << 31) - 1\n"
    "_U32_OFF = __import__(\"os\").environ.get(\"KF_SPARSE_U32_OFF\", \"1\") == \"1\"\n",
)
open(dst, "w").write(s)
print("wrote", dst)
