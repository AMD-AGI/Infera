#!/usr/bin/env python3
"""Build a KF sparse MLA variant whose split-K decode kernel gathers a > 2 GiB KV pool without masked loads.

Why the production kernel is ~30% slower above 2 GiB (ISA, 11-11): Triton's HIP backend gives up buffer loads for a
pointer whose storage exceeds 2^31 - 1 bytes, and a *masked* global load is lowered with an exec-mask branch around it
(18 loads -> 54 s_and_saveexec / s_cbranch_execz), which splits the software-pipelined loop body into small blocks;
buffer loads mask through out-of-range offsets and need none. The 64-bit address math itself costs nothing measurable
(scalar-base 32-bit offsets: same time). Invalid rows already point at slot 0 (`page = where(valid, slot, 0)`), so the
loads can go unmasked, and the invalid rows are zeroed afterwards -- exactly what the masked load returned, whatever
slot 0 holds.

Usage: python3 make_kf_sparse_nomask.py SRC DST ADDR ZERO
    ADDR: u32 (pool base + one zero-extended 32-bit offset, pools <= 4 GiB) | i64 (production's 64-bit rows)
    ZERO: and (bitwise AND with a per-row byte mask; 1-byte KV only, else falls back to where) | where (select)
"""
import sys

src, dst, addr, zero = sys.argv[1:5]
assert addr in ("u32", "i64") and zero in ("and", "where")
s = open(src).read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"anchor found {s.count(old)} times, want {count}: {old[:80]!r}"
    s = s.replace(old, new)


sub(
    "def _page_offsets_fit_i32(num_pages: int, kv_dim: int) -> bool:\n",
    '''@triton.jit
def _gather_rows_nomask(kv_ptr, row, col, valid, CACHE_MOD: tl.constexpr, U32: tl.constexpr, ZERO_AND: tl.constexpr):
    """KV[row + col] for every row, unmasked (invalid rows point at slot 0), with the invalid rows zeroed."""
    if U32:
        x = tl.load(kv_ptr + (row + col.to(tl.uint32)), cache_modifier=CACHE_MOD)
    else:
        x = tl.load(kv_ptr + (row + col), cache_modifier=CACHE_MOD)
    if ZERO_AND:
        keep = tl.where(valid, 255, 0).to(tl.uint8)
        return (x.to(tl.uint8, bitcast=True) & keep[:, None]).to(x.dtype, bitcast=True)
    return tl.where(valid[:, None], x, 0.0)


def _page_offsets_fit_i32(num_pages: int, kv_dim: int) -> bool:
''',
)
sub(
    """    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
):
    \"\"\"Split-K partial kernel. grid=(seq, head_blocks, kv_splits).
""",
    """    PARTIAL_STORE_MOD: tl.constexpr = "",
    FIXED_TRIP: tl.constexpr = False,
    NOMASK_GATHER: tl.constexpr = False,
    NOMASK_U32: tl.constexpr = False,
    NOMASK_AND: tl.constexpr = False,
):
    \"\"\"Split-K partial kernel. grid=(seq, head_blocks, kv_splits).

    ``NOMASK_GATHER`` (pools past 2 GiB, which lose buffer loads): gather the KV rows with unmasked global loads and
    zero the invalid rows afterwards; a masked global load is lowered with an exec-mask branch that breaks up the
    pipelined loop. ``NOMASK_U32`` addresses with one zero-extended 32-bit offset (pools <= 4 GiB) instead of 64-bit
    rows; ``NOMASK_AND`` zeroes with a byte AND (1-byte KV) instead of a select.
""",
)
old_gather = """        page = tl.where(valid, slot, 0)
        if USE_I64_PAGE:
            kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
        else:
            kv_base = kv_ptr + page[:, None] * KV_DIM

        kv0 = tl.load(
"""
new_gather = """        page = tl.where(valid, slot, 0)
        if NOMASK_GATHER:
            if NOMASK_U32:
                kv_row = page[:, None].to(tl.uint32) * KV_DIM
            else:
                kv_row = page[:, None].to(tl.int64) * KV_DIM
            kv0 = _gather_rows_nomask(
                kv_ptr, kv_row, g[None, :], valid, KV_CACHE_MOD, NOMASK_U32, NOMASK_AND
            ).to(input_type)
            if NUM_GROUPS >= 2:
                kv1 = _gather_rows_nomask(
                    kv_ptr, kv_row, (GW + g)[None, :], valid, KV_CACHE_MOD, NOMASK_U32, NOMASK_AND
                ).to(input_type)
            if NUM_GROUPS >= 3:
                kv2 = _gather_rows_nomask(
                    kv_ptr, kv_row, (2 * GW + g)[None, :], valid, KV_CACHE_MOD, NOMASK_U32, NOMASK_AND
                ).to(input_type)
            if NUM_GROUPS >= 4:
                kv3 = _gather_rows_nomask(
                    kv_ptr, kv_row, (3 * GW + g)[None, :], valid, KV_CACHE_MOD, NOMASK_U32, NOMASK_AND
                ).to(input_type)
            kv_tail = _gather_rows_nomask(
                kv_ptr, kv_row, (D_V + dt)[None, :], valid, KV_CACHE_MOD, NOMASK_U32, NOMASK_AND
            ).to(input_type)
        elif USE_I64_PAGE:
            kv_base = kv_ptr + page[:, None].to(tl.int64) * KV_DIM
        else:
            kv_base = kv_ptr + page[:, None] * KV_DIM

        if not NOMASK_GATHER:
            kv0 = tl.load(
"""
sub(old_gather, new_gather)
# the rest of the original gather moves under `if not NOMASK_GATHER:`
start = s.index("        if not NOMASK_GATHER:\n            kv0 = tl.load(\n")
end = s.index("        scores = tl.dot(q0, tl.trans(kv0))", start)
head, rest = s[start:end].split("            kv0 = tl.load(\n", 1)
rest = "".join(("    " + ln if ln.strip() else ln) for ln in rest.splitlines(True))
s = s[:start] + head + "            kv0 = tl.load(\n" + rest + s[end:]

u32_cond = "kv.shape[0] * kv.stride(0) * kv.element_size() <= (1 << 32) - 1" if addr == "u32" else "False"
sub(
    "    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)\n",
    "    split_i32_page = decode_tuned and _page_offsets_fit_i32(kv.shape[0], kv_dim)\n"
    "    split_nomask = bool(decode_tuned) and _NOMASK and not split_i32_page and kv.is_contiguous()\n"
    f"    split_nomask_u32 = split_nomask and {u32_cond}\n"
    f"    split_nomask_and = split_nomask and {'kv.element_size() == 1' if zero == 'and' else 'False'}\n",
)
sub(
    """            FIXED_TRIP=split_fixed_trip,
            num_warps=split_num_warps,
""",
    """            FIXED_TRIP=split_fixed_trip,
            NOMASK_GATHER=split_nomask,
            NOMASK_U32=split_nomask_u32,
            NOMASK_AND=split_nomask_and,
            num_warps=split_num_warps,
""",
)
sub(
    "_I32_MAX = (1 << 31) - 1\n",
    "_I32_MAX = (1 << 31) - 1\n"
    "_NOMASK = __import__(\"os\").environ.get(\"KF_SPARSE_NOMASK_GATHER\", \"1\") == \"1\"\n",
)
open(dst, "w").write(s)
print("wrote", dst, addr, zero)
