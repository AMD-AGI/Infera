# mk_pack2 (task dcpx10 2026-10-22): write dcp_pack.yihou-nowait2.py = dcp_pack.yihou-nowait.py (dcpx9) plus, when
# YIHOU_PACK_NOWAIT=2: (a) per-layer metadata built once per (ptrs, widths, strides) and kept on the device (the stock
# copy_mla_rows_into_pack does torch.tensor(metadata, device=...) = a pageable H2D on every call), offsets scaled by n
# inside the kernel; (b) the gather runs on a high-priority stream (priority -1) so it is not queued behind the prefill
# compute kernels. dcpx9 sub-timers: launch 35 ms + sync 36 ms per pack, index copy 0.15 ms.
# Usage: python3 -I mk_pack2.yihou.py <dcp_pack.yihou-nowait.py> <dst>
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = """    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
    gather_stream = pack_buffer.get_gather_stream()
    _t0 = time.perf_counter()
    if _YIHOU_PACK_NOWAIT:
"""
new = """    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
    gather_stream = pack_buffer.get_gather_stream()
    _t0 = time.perf_counter()
    if _YIHOU_PACK_NOWAIT_MODE == "2":
        # yihou dcpx10: cached device metadata + high-priority gather stream.
        hp = getattr(pack_buffer, "_yh_hp_stream", None)
        if hp is None:
            hp = torch.cuda.Stream(device=pack.device, priority=-1)
            pack_buffer._yh_hp_stream = hp
        gather_stream = hp
        with torch.cuda.stream(hp):
            row_indices = (
                torch.from_numpy(np.ascontiguousarray(src_token_indices, dtype=np.int64))
                .pin_memory()
                .to(pack.device, non_blocking=True)
            )
            _t1 = time.perf_counter()
            _yh_copy_rows_cached(
                kv_data_ptrs, row_indices, pack, token_item_lens, src_token_item_lens
            )
    elif _YIHOU_PACK_NOWAIT:
"""
assert s.count(old) == 1, "anchor1"
s = s.replace(old, new)
a = '_YIHOU_PACK_NOWAIT = os.environ.get("YIHOU_PACK_NOWAIT") == "1"\n'
assert s.count(a) == 1, "anchor2"
s = s.replace(a, '''_YIHOU_PACK_NOWAIT_MODE = os.environ.get("YIHOU_PACK_NOWAIT", "")
_YIHOU_PACK_NOWAIT = _YIHOU_PACK_NOWAIT_MODE == "1"

import triton
import triton.language as tl


@triton.jit
def _yh_pack_rows_kernel(
    src_metadata,
    row_indices,
    pack,
    num_rows,
    BLOCK_SIZE: tl.constexpr,
):
    # Same as sglang kernels/ops/kvcache/pd_dcp_gather.py, except metadata[2] is the per-row byte prefix and the
    # pack offset is prefix * num_rows, so the metadata does not depend on n and can stay on the device.
    layer_id = tl.program_id(0)
    block_id = tl.program_id(1)
    metadata_offset = layer_id * 4
    src = tl.load(src_metadata + metadata_offset).to(pack.dtype)
    row_nbytes = tl.load(src_metadata + metadata_offset + 1)
    pack_offset = tl.load(src_metadata + metadata_offset + 2) * num_rows
    src_row_stride = tl.load(src_metadata + metadata_offset + 3)

    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    layer_nbytes = num_rows * row_nbytes
    mask = offsets < layer_nbytes
    row = offsets // row_nbytes
    byte = offsets % row_nbytes
    src_row = tl.load(row_indices + row, mask=mask, other=0)
    values = tl.load(src + src_row * src_row_stride + byte, mask=mask)
    tl.store(pack + pack_offset + offsets, values, mask=mask)


_YH_META_CACHE = {}


def _yh_copy_rows_cached(kv_data_ptrs, row_indices, pack, token_item_lens, src_token_item_lens=None):
    if src_token_item_lens is None:
        src_token_item_lens = token_item_lens
    if not kv_data_ptrs:
        return
    key = (str(pack.device), tuple(int(p) for p in kv_data_ptrs), tuple(int(x) for x in token_item_lens),
           tuple(int(x) for x in src_token_item_lens))
    meta = _YH_META_CACHE.get(key)
    if meta is None:
        rows = []
        prefix = 0
        for ptr, item_len, src_item_len in zip(key[1], key[2], key[3]):
            if item_len <= 0:
                raise ValueError(f"MLA token item length must be positive, got {item_len}")
            rows.extend((ptr, item_len, prefix, src_item_len))
            prefix += item_len
        # Built once per key; this H2D is not on the hot path.
        meta = torch.tensor(rows, dtype=torch.int64, device=pack.device)
        _YH_META_CACHE[key] = meta
    n = int(row_indices.numel())
    grid = (len(key[1]), triton.cdiv(n * max(key[2]), 1024))
    _yh_pack_rows_kernel[grid](meta, row_indices, pack, n, BLOCK_SIZE=1024)
''')
s = s.replace("# mk_pack (task dcpx9", "# mk_pack (task dcpx9")
open(dst, "w").write(s)
print("ok", dst)
