from typing import Optional

import torch

# ---------------------------------------------------------------------------------------------
# YIHOU-SKGUARD (idxshare debug 2026-10-20). The decode "Memory access fault" is a logits read
# past the end of the score buffer inside sgl::dsa_topk::coop_topk_kernel (debug agent, twice),
# with row length 60k-99k against a score buffer of < ~46 KB. The Indexer-level guards (v1-v4)
# never counted a bad row, so this guard sits at the choke point every Python caller of the AOT
# top-k ops passes through, and keys its counters by the CALL SITE (3 frames, recorded at eager
# call / graph capture time). Counters live on the device (no host sync on the hot path; graph
# replays update them) and are printed every 60 s by a daemon thread.
#   YIHOU_TOPK_GUARD = off | count | mask   (mask clamps lengths into the score row)
# ---------------------------------------------------------------------------------------------
import logging as _yg_logging
import os as _yg_os
import sys as _yg_sys
import threading as _yg_threading
import time as _yg_time

_YG_MODE = _yg_os.environ.get("YIHOU_SKGUARD") or _yg_os.environ.get("YIHOU_TOPK_GUARD", "off")
_YG_MAX_SITES = 256
_YG_SITES = {}
_YG_CTR = {}
_YG_LOCK = _yg_threading.Lock()
_YG_LAST = {}
_yg_logger = _yg_logging.getLogger("sglang.yihou_skguard")


def _yg_site():
    f = _yg_sys._getframe(1)
    for _ in range(2):  # skip _yg_guard and the fast_topk_* wrapper
        f = f.f_back if f is not None else None
    parts = []
    for _ in range(4):
        if f is None:
            break
        parts.append(
            f"{_yg_os.path.basename(f.f_code.co_filename)}:{f.f_code.co_name}:{f.f_lineno}"
        )
        f = f.f_back
    return "<".join(parts)


def _yg_log_loop(dev, ctr):
    torch.cuda.set_device(dev)
    st = torch.cuda.Stream(device=dev)
    while True:
        _yg_time.sleep(60)
        with torch.cuda.stream(st):
            vals = ctr.to("cpu").tolist()
        with _YG_LOCK:
            sites = sorted(_YG_SITES.items(), key=lambda kv: kv[1])
        lines = []
        for key, i in sites:
            v = vals[i]
            if v[0]:
                lines.append(
                    f"[{i}] {key} calls={v[0]} rows={v[1]} bad_rows={v[2]} bad_calls={v[3]} "
                    f"max_over={v[4]} last_width={v[5]} max_len={v[6]}"
                )
        _yg_logger.warning("YIHOU-SKGUARD mode=%s dev=%s\n  %s", _YG_MODE, dev, "\n  ".join(lines))


def _yg_dump_batch():
    """v3: on a BAD eager call, log the ForwardBatch / attention metadata the caller saw."""
    try:
        f = _yg_sys._getframe(1)
        fb = None
        meta = None
        depth = 0
        while f is not None and depth < 60:
            loc = f.f_locals
            if fb is None and "forward_batch" in loc:
                fb = loc["forward_batch"]
            if meta is None and "metadata" in loc and hasattr(loc["metadata"], "attn_metadata"):
                meta = loc["metadata"]
            f = f.f_back
            depth += 1

        def t(x, n=16):
            if x is None:
                return None
            try:
                return x.flatten()[:n].tolist()
            except Exception:  # noqa: BLE001
                return repr(x)[:200]

        info = {}
        if fb is not None:
            info.update(
                mode=str(getattr(fb, "forward_mode", None)),
                orig_mode=str(getattr(fb, "_original_forward_mode", None)),
                bs=getattr(fb, "batch_size", None),
                seq_lens=t(getattr(fb, "seq_lens", None)),
                seq_lens_cpu=t(getattr(fb, "seq_lens_cpu", None)),
                seq_lens_sum=getattr(fb, "seq_lens_sum", None),
                req_pool_indices=t(getattr(fb, "req_pool_indices", None)),
                extend_seq_lens_cpu=getattr(fb, "extend_seq_lens_cpu", None),
                num_token_non_padded_cpu=getattr(fb, "num_token_non_padded_cpu", None),
                global_num_tokens_cpu=getattr(fb, "global_num_tokens_cpu", None),
                spec_info=type(getattr(fb, "spec_info", None)).__name__,
                capture_hidden_mode=str(getattr(fb, "capture_hidden_mode", None)),
                input_ids_shape=tuple(getattr(fb, "input_ids").shape) if getattr(fb, "input_ids", None) is not None else None,
            )
        if meta is not None:
            am = meta.attn_metadata
            info.update(
                am_cache_seqlens=t(getattr(am, "cache_seqlens_int32", None)),
                am_seqlens_expanded=t(getattr(am, "dsa_seqlens_expanded", None)),
                am_max_seq_len_k=getattr(am, "max_seq_len_k", None),
                am_page_table_1_shape=tuple(am.page_table_1.shape) if getattr(am, "page_table_1", None) is not None else None,
            )
        _yg_logger.warning("YIHOU-SKGUARD-EAGER BAD batch=%s", info)
    except Exception as e:  # noqa: BLE001
        _yg_logger.warning("YIHOU-SKGUARD-EAGER BAD batch dump failed: %r", e)


def _yg_ensure(dev):
    """Allocate this device's counters (+ logger thread) outside any graph capture. Called by
    the mounted dsa_indexer (Indexer.__init__) so graphs captured before the first eager top-k
    call carry the guard too."""
    if _YG_MODE == "off":
        return None
    with _YG_LOCK:
        ctr = _YG_CTR.get(dev.index)
        if ctr is not None:
            return ctr
        # cols: calls, rows, bad_rows, calls_with_bad, max(end - width), width at last call, max(len)
        ctr = torch.zeros((_YG_MAX_SITES, 7), dtype=torch.int64, device=dev)
        _YG_CTR[dev.index] = ctr
    _yg_threading.Thread(
        target=_yg_log_loop, args=(dev, ctr), daemon=True, name="yihou-skguard"
    ).start()
    return ctr


def _yg_guard(fn_name, score, lengths, row_starts):
    if _YG_MODE == "off" or lengths is None:
        return lengths
    capturing = torch.cuda.is_current_stream_capturing()
    dev = score.device
    ctr = _YG_CTR.get(dev.index)
    if ctr is None:
        if capturing:
            return lengths
        ctr = _yg_ensure(dev)
    key = f"{fn_name}|{'graph' if capturing else 'eager'}|{_yg_site()}"
    with _YG_LOCK:
        i = _YG_SITES.get(key)
        if i is None:
            if len(_YG_SITES) >= _YG_MAX_SITES:
                return lengths
            i = _YG_SITES[key] = len(_YG_SITES)
    width = score.shape[1]
    ln = lengths.to(torch.int64)
    end = ln + row_starts.to(torch.int64) if row_starts is not None else ln
    bad = (end > width) | (ln < 0)
    nbad = bad.sum()
    row = ctr[i]
    row[0].add_(1)
    row[1].add_(lengths.shape[0])
    row[2].add_(nbad)
    row[3].add_((nbad > 0).to(torch.int64))
    if ln.numel() > 0:
        row[4].copy_(torch.maximum(row[4], (end - width).max()))
        row[6].copy_(torch.maximum(row[6], ln.max()))
    row[5].fill_(width)
    if not capturing:
        # v2: eager calls are rare in serving and the faulting call kills the process before
        # the periodic logger runs, so check eager calls synchronously and log a bad one (and
        # every site at most once per 30 s) BEFORE the kernel is launched.
        nb = int(nbad)
        now = _yg_time.time()
        last = _YG_LAST.get(i, 0.0)
        if nb or now - last > 30:
            _YG_LAST[i] = now
            _yg_logger.warning(
                "YIHOU-SKGUARD-EAGER%s %s score=%s lengths=%s width=%d max_len=%d min_len=%d nbad=%d row_starts=%s",
                " BAD" if nb else "",
                key,
                tuple(score.shape),
                tuple(lengths.shape),
                width,
                int(ln.max()) if ln.numel() else -1,
                int(ln.min()) if ln.numel() else -1,
                nb,
                None if row_starts is None else tuple(row_starts.shape),
            )
            if nb:
                bad_idx = torch.nonzero(bad).flatten()[:8].tolist()
                _yg_logger.warning(
                    "YIHOU-SKGUARD-EAGER BAD rows=%s lens=%s stack=%s",
                    bad_idx,
                    [int(ln[j]) for j in bad_idx],
                    "".join(__import__("traceback").format_stack(limit=40)[:-1]),
                )
                _yg_dump_batch()
    if _YG_MODE == "mask":
        if row_starts is not None:
            return torch.minimum(lengths, width - row_starts.to(lengths.dtype)).clamp_min(0)
        return lengths.clamp(0, width)
    return lengths



def fast_topk(values, topk, dim):
    if topk == 1:
        # Use max along the specified dimension to get both value and index
        return torch.max(values, dim=dim, keepdim=True)
    else:
        # Use topk for efficiency with larger k values
        # TODO: implement faster cuda kernels for large vocab sizes
        return torch.topk(values, topk, dim=dim)


def fast_topk_v2(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert topk == 2048, (
        "fast_topk_v2 is only optimized for deepseek v3.2 model, where topk=2048"
    )
    assert score.dim() == 2
    topk_indices = score.new_empty((score.size(0), topk), dtype=torch.int32)
    lengths = _yg_guard("fast_topk_v2", score, lengths, row_starts)
    torch.ops.sgl_kernel.fast_topk(score, topk_indices, lengths, row_starts)
    return topk_indices


def fast_topk_transform_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    page_table_size_1: torch.Tensor,  # NOTE: page size should be 1
    cu_seqlens_q: torch.Tensor,
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices
    to indices to the page table (page_size = 1)
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key whose layout is either ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        page_table_size_1: The page table tensor of shape (Batch, topk)
        cu_seqlens_q: The cumulative sequence lengths tensor of shape (Batch + 1)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It's only used for cases where the key is
            ragged, i.e. during extend and draft extend.
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert topk == 2048, (
        "fast_topk_transform_fused is only optimized for deepseek v3.2 model, where topk=2048"
    )
    assert score.dim() == 2
    src_page_table = page_table_size_1
    dst_page_table = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    lengths = _yg_guard("fast_topk_transform_fused", score, lengths, row_starts)
    torch.ops.sgl_kernel.fast_topk_transform_fused(
        score, lengths, dst_page_table, src_page_table, cu_seqlens_q, row_starts
    )
    return dst_page_table


def deepseek_v4_topk_transform_512(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_indices: torch.Tensor,
    page_size: int,
    raw_indices: Optional[torch.Tensor] = None,
    sort_output: bool = False,
) -> None:
    """
    Performs the DeepSeek-V4 indexer top-k selection and writes the paged
    physical slot indices into ``page_indices``. Supports topk up to 1024.
    Optionally also writes the row-relative raw token positions into
    ``raw_indices`` for hisparse capture.

    Args:
        scores: float32 ``[B, max_seq_len]`` indexer logits, contiguous on dim 1.
        seq_lens: int32 ``[B]``, true KV length per batch row; each at most
            ``max_seq_len`` and ``num_pages * page_size``.
        page_table: int32 ``[B, num_pages]``, logical->physical page table,
            contiguous on dim 1.
        page_indices: int32 ``[B, topk]``, output buffer, contiguous. Filled
            with paged physical slots; -1 for padding entries.
        page_size: power-of-2 page size.
        raw_indices: optional int32 ``[B, topk]``, contiguous. If provided,
            filled with raw token positions within each row.
        sort_output: sort every row ascending in the kernel epilogue, with the
            -1 padding last: by raw position when ``raw_indices`` is given,
            otherwise by physical slot. The selected set is unchanged.
    """
    if raw_indices is not None:
        assert raw_indices.dim() == 2
    torch.ops.sgl_kernel.deepseek_v4_topk_transform_512(
        scores, seq_lens, page_table, page_indices, page_size, raw_indices, sort_output
    )


def fast_topk_transform_ragged_fused(
    score: torch.Tensor,
    lengths: torch.Tensor,
    topk_indices_offset: torch.Tensor,  # ragged kv
    topk: int,
    row_starts: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Get the topk indices of the score tensor and then transform the topk indices to
    indices to ragged kv (non-paged). This function is only used for extend,
    not including draft extend.
    Args:
        score: The score tensor of shape (B, L). The score tensor is the logits
            between the query and the key which can be ragged or paged.
            row_starts is only required when the key is ragged.
        lengths: The lengths tensor of shape (B)
        topk_indices_offset: The offset of topk indices in ragged kv of shape (B)
        topk: The number of topk indices to get
        row_starts: The start index of each row in the score tensor of shape (B).
            For each row i, topk only applies to section [row_starts[i], row_starts[i] + lengths[i]]
            of the score tensor. It can be None if only the fast path is triggered,
            in the case of all values in lengths <= topk (not checked in the kernel,
            guaranteed by the caller).
    Returns:
        The topk indices tensor of shape (B, topk)
    """
    assert topk == 2048, (
        "fast_topk_transform_ragged_fused is only optimized for deepseek v3.2 model, where topk=2048"
    )
    assert score.dim() == 2
    topk_indices_ragged = score.new_empty((score.shape[0], topk), dtype=torch.int32)
    lengths = _yg_guard("fast_topk_transform_ragged_fused", score, lengths, row_starts)
    torch.ops.sgl_kernel.fast_topk_transform_ragged_fused(
        score, lengths, topk_indices_ragged, topk_indices_offset, row_starts
    )
    return topk_indices_ragged
