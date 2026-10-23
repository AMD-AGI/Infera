# mk_pack (task dcpx9 2026-10-22): write dcp_pack.yihou-nowait.py = overlay 93034eba91 common/dcp_pack.py with
# (a) sub-timers YH_T = [t_index, t_launch, t_sync, n] (read and reset by the [yihou-xfer] stats in conn.py), and
# (b) YIHOU_PACK_NOWAIT=1: row indices copied pinned + non_blocking ON the gather stream, and NO wait on the device
#     default stream. Ordering vs the prefill forward comes from TransferKVChunk.wait_event (synchronized in
#     transfer_worker before the send) or from the chunk being queued after the forward resolved (same reasoning as
#     upstream staging_buffer.producer_stream docstring, sglang 8e3e2980db / #40861).
# Usage: python3 -I mk_pack.yihou.py <src dcp_pack.py> <dst>
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src).read()
old = """    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
    row_indices = torch.as_tensor(
        src_token_indices, device=pack.device, dtype=torch.int64
    )
    gather_stream = pack_buffer.get_gather_stream()
    gather_stream.wait_stream(torch.cuda.default_stream(pack.device))
    with torch.cuda.stream(gather_stream):
        copy_mla_rows_into_pack(
            kv_data_ptrs, row_indices, pack, token_item_lens, src_token_item_lens
        )
    gather_stream.synchronize()
"""
new = """    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
    gather_stream = pack_buffer.get_gather_stream()
    _t0 = time.perf_counter()
    if _YIHOU_PACK_NOWAIT:
        # yihou dcpx9: keep the pack off the device default stream (no pageable H2D on it, no wait on it).
        with torch.cuda.stream(gather_stream):
            row_indices = (
                torch.from_numpy(np.ascontiguousarray(src_token_indices, dtype=np.int64))
                .pin_memory()
                .to(pack.device, non_blocking=True)
            )
            _t1 = time.perf_counter()
            copy_mla_rows_into_pack(
                kv_data_ptrs, row_indices, pack, token_item_lens, src_token_item_lens
            )
    else:
        row_indices = torch.as_tensor(
            src_token_indices, device=pack.device, dtype=torch.int64
        )
        _t1 = time.perf_counter()
        gather_stream.wait_stream(torch.cuda.default_stream(pack.device))
        with torch.cuda.stream(gather_stream):
            copy_mla_rows_into_pack(
                kv_data_ptrs, row_indices, pack, token_item_lens, src_token_item_lens
            )
    _t2 = time.perf_counter()
    gather_stream.synchronize()
    _t3 = time.perf_counter()
    YH_T[0] += _t1 - _t0
    YH_T[1] += _t2 - _t1
    YH_T[2] += _t3 - _t2
    YH_T[3] += 1
"""
assert s.count(old) == 1, "pack anchor"
s = s.replace(old, new)
a = "logger = logging.getLogger(__name__)\n"
assert s.count(a) == 1
s = s.replace(a, a + '\nimport os\nimport time\n\n_YIHOU_PACK_NOWAIT = os.environ.get("YIHOU_PACK_NOWAIT") == "1"\n'
              '# yihou dcpx9: [t_index, t_launch, t_sync, n] summed over try_pack_dcp_src calls (racy +=, log-only).\nYH_T = [0.0, 0.0, 0.0, 0]\n')
open(dst, "w").write(s)
print("ok", dst)
