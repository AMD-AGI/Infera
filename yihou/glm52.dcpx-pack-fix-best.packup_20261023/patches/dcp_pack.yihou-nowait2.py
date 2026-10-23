from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

from sglang.kernels.ops.kvcache.pd_dcp_gather import copy_mla_rows_into_pack
from sglang.srt.disaggregation.common.staging_buffer import StagingBuffer

logger = logging.getLogger(__name__)

import os
import time

_YIHOU_PACK_NOWAIT_MODE = os.environ.get("YIHOU_PACK_NOWAIT", "")
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
# yihou dcpx9: [t_index, t_launch, t_sync, n] summed over try_pack_dcp_src calls (racy +=, log-only).
YH_T = [0.0, 0.0, 0.0, 0]


def dcp_pack_buffer_bytes(
    kv_item_lens: Sequence[int], page_size: int, max_tokens: int, dcp_size: int = 1
) -> int:
    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if dcp_size <= 0:
        raise ValueError(f"dcp_size must be positive, got {dcp_size}")
    if any(item_len < page_size for item_len in kv_item_lens):
        raise ValueError(
            "PD DCP pack requires each kv_item_len to span at least one page, "
            f"got {list(kv_item_lens)} with page_size={page_size}"
        )
    if any(item_len % page_size != 0 for item_len in kv_item_lens):
        raise ValueError(
            "PD DCP pack requires page-aligned kv_item_lens, "
            f"got {list(kv_item_lens)} with page_size={page_size}"
        )
    token_item_lens = [item_len // page_size for item_len in kv_item_lens]
    rank_tokens = (max_tokens + dcp_size - 1) // dcp_size
    return dcp_size * rank_tokens * sum(token_item_lens)


def try_pack_dcp_src(
    *,
    pack_buffer: StagingBuffer,
    kv_data_ptrs: Sequence[int],
    src_token_indices: npt.NDArray[np.integer],
    token_item_lens: Sequence[int],
    src_token_item_lens: Optional[Sequence[int]] = None,
    pack_offset_bytes: int = 0,
    pack_capacity_bytes: Optional[int] = None,
) -> Optional[Tuple[List[int], npt.NDArray[np.int64]]]:
    if pack_offset_bytes < 0:
        raise ValueError(
            f"pack_offset_bytes must be non-negative, got {pack_offset_bytes}"
        )
    n = int(src_token_indices.size)
    if n == 0:
        empty = np.empty((0,), dtype=np.int64)
        return [], empty
    required = n * sum(int(item_len) for item_len in token_item_lens)
    required_end = pack_offset_bytes + required
    if (
        pack_capacity_bytes is not None and required > pack_capacity_bytes
    ) or not pack_buffer.fits(required_end):
        logger.warning(
            "PD DCP pack buffer too small for byte range [%s, %s) "
            "(have %s, region capacity %s); "
            "falling back to per-token RDMA",
            pack_offset_bytes,
            required_end,
            pack_buffer.get_size(),
            pack_capacity_bytes,
        )
        return None

    pack = pack_buffer.buffer.narrow(0, pack_offset_bytes, required)
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

    packed_ptrs: List[int] = []
    offset = 0
    base = pack_buffer.get_ptr() + pack_offset_bytes
    for item_len in token_item_lens:
        packed_ptrs.append(base + offset)
        offset += n * int(item_len)
    return packed_ptrs, np.arange(n, dtype=np.int64)


def init_dcp_pack_buffers(
    register_fn,
    kv_args,
    count: int,
    dcp_size: int,
    max_tokens: int,
    *,
    include_draft: bool = False,
) -> List[StagingBuffer]:
    from sglang.srt.disaggregation.common.staging_handler import (
        _get_custom_mem_pool,
    )

    kv_item_lens = kv_args.kv_item_lens
    if not include_draft and kv_args.num_draft_entries:
        kv_item_lens = kv_item_lens[: -kv_args.num_draft_entries]
    # Note(kpham-sgl): size = dcp_size x ceil(max_tokens / dcp_size)
    # x sum(per-layer token bytes). At 32,768 tokens and 61 MLA layers
    # x 576 bf16 dims x 2 B: 2.14 GiB/buffer, 8.58 GiB for 4 queues.
    size_bytes = dcp_pack_buffer_bytes(
        kv_item_lens, kv_args.page_size, max_tokens, dcp_size
    )
    gpu_id = kv_args.gpu_id
    device = f"cuda:{gpu_id}"
    custom_mem_pool, _ = _get_custom_mem_pool(device)

    buffers = []
    for _ in range(count):
        buf = StagingBuffer(size_bytes, device, gpu_id, custom_mem_pool=custom_mem_pool)
        register_fn(buf.get_ptr(), buf.get_size())
        buffers.append(buf)
    logger.info(
        "PD DCP pack buffers allocated: %d x %.1f MB (max_tokens=%d)",
        count,
        size_bytes / (1024 * 1024),
        max_tokens,
    )
    return buffers
