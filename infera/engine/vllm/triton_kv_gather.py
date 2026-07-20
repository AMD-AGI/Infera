###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
"""Triton kernels for vLLM-8 chunked-fusion KV cache scatter/gather.

Pattern: a v2 chunk's payload lives in a contiguous staging buffer of
shape ``[2, num_layers, chunk_tokens, hidden_dim]`` (the
``KV_2LTD`` format defined in `packed_format.ChunkHeader`).

On LOAD we scatter that staging tensor into per-layer paged KV
caches: for each `token_idx_in_chunk`, the destination block_id
within the layer's `[2, num_blocks, block_size, num_kv_heads,
head_dim]` tensor is `slot_mapping[token_idx_in_chunk]` (precomputed
as `page_id_for_token * block_size + token_offset_in_page` by the
caller, so the kernel itself sees one int per token).

On SAVE we gather in the opposite direction.

Both directions use the same kernel with a `DIRECTION: tl.constexpr`
flag so the compiler specializes — no runtime branching in the inner
loop. Layout fits Triton's strong points (contiguous tile loads,
masked writes) and avoids any warp-level intrinsic / inline PTX,
which keeps the kernel portable across CUDA and ROCm via the same
Triton source.

A Python-loop fallback (`kv_chunk_scatter_python`,
`kv_chunk_gather_python`) is provided for: (a) unit-test correctness
oracle, (b) hardware-fallback when Triton isn't available on the
worker. The fallback uses `slot.copy_(staging)` per (layer, page) —
slower (re-introduces the per-call overhead the chunk design exists
to avoid) but functionally equivalent.

Design references (read for layout / mapping inspiration; nothing
copied):
  - LMCache `csrc/mem_kernels.cu::load_and_reshape_multi_layer_kernel`
    (Apache-2.0). Our kernel matches its 3D
    `(num_tokens, num_layers, k_or_v)` decomposition because that's
    the natural shape of the problem; the inner code is ours.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import torch

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Triton availability probe — triton.jit is the only thing we use; if
# importing it fails (e.g. very old PyTorch wheel, custom build), the
# caller transparently falls back to the Python loop. The probe is
# module-level so the cost is paid once at process import time.
# ----------------------------------------------------------------------

try:
    import triton  # noqa: F401
    import triton.language as tl

    _TRITON_AVAILABLE = True
except ImportError:  # pragma: no cover — triton is in modern torch
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    """``True`` if Triton kernels can run on this worker; ``False``
    if the caller must use the Python-loop fallback."""
    return _TRITON_AVAILABLE


# ----------------------------------------------------------------------
# Triton kernel (load+save in one body via the DIRECTION flag)
# ----------------------------------------------------------------------


if _TRITON_AVAILABLE:

    @triton.jit
    def _kv_chunk_transfer_kernel(
        # Staging buffer: [2, num_layers, chunk_tokens, hidden_dim] flat
        staging_ptr,
        # int64 array, length num_layers — each entry is one paged-KV
        # tensor's data_ptr() (cast to int64 by caller). Per-layer
        # pointers, not one big concat, because vLLM allocates per-layer
        # caches as independent torch.empty calls.
        layer_ptrs,
        # int32 array, length chunk_tokens — destination "slot" within
        # the layer's paged buffer for each token_idx_in_chunk. Caller
        # computes slot = block_id * block_size + token_offset_in_page,
        # so the kernel only does one addressing step. For HMA models
        # we pass ONE slot_mapping per cache_group (one chunk-key per
        # group), so block_id selection is already resolved.
        slot_mapping_ptr,
        # Strides (in ELEMENTS, not bytes — Triton convention)
        layer_k_stride: tl.constexpr,  # K-base → V-base distance per layer (ignored when NUM_KV_CHANNELS=1)
        layer_slot_stride: tl.constexpr,  # slot N → slot N+1 distance per layer (= hidden_dim)
        # Sizes
        num_layers: tl.constexpr,
        chunk_tokens: tl.constexpr,
        hidden_dim: tl.constexpr,
        block_size: tl.constexpr,  # tokens per page — only used for the KV_INTERLEAVED block-gap term
        NUM_KV_CHANNELS: tl.constexpr,  # 2 = K+V split (regular attention); 1 = MLA (single combined latent)
        KV_INTERLEAVED: tl.constexpr,  # 0 = K/V outermost [2, nb, bs, H]; 1 = per-block interleaved [nb, 2, bs, H] (vLLM 0.23 FullAttention)
        DIRECTION: tl.constexpr,  # 0 = LOAD (staging→layer); 1 = SAVE (layer→staging)
        TOKEN_BLOCK: tl.constexpr,  # tokens per Triton program (tiling)
    ):
        """One program handles ``TOKEN_BLOCK`` consecutive tokens for
        ONE (kv_idx, layer_idx) pair. Grid = ``(NUM_KV_CHANNELS × num_layers,
        ceil(chunk_tokens / TOKEN_BLOCK))``.

        For NUM_KV_CHANNELS=2 (regular attention) kv_idx ∈ {0,1} selects
        K vs V at base offset 0 / layer_k_stride within the layer
        tensor. For NUM_KV_CHANNELS=1 (MLA) kv_idx is always 0 and
        layer_k_stride is unused — the layer tensor is a single
        combined K+V latent of shape [num_blocks, block_size, hidden_dim].

        Per token, we transfer ``hidden_dim`` elements between a
        contiguous staging row and a (possibly far away) per-layer
        paged-KV row. Both addresses are computed from indices, so
        the inner loop is just a masked load + masked store.
        """
        kv_layer_id = tl.program_id(0)
        if NUM_KV_CHANNELS == 1:
            layer_idx = kv_layer_id
            kv_idx = 0
        else:
            layer_idx = kv_layer_id // 2
            kv_idx = kv_layer_id % 2  # 0 = K, 1 = V
        token_block_id = tl.program_id(1)

        # Per-thread token index within the chunk
        token_offsets = token_block_id * TOKEN_BLOCK + tl.arange(0, TOKEN_BLOCK)
        token_mask = token_offsets < chunk_tokens

        # Resolve per-layer destination pointer (paged KV for this
        # layer). Cast int64 → pointer; Triton handles the typing.
        layer_base = tl.load(layer_ptrs + layer_idx)
        layer_ptr = tl.cast(layer_base, tl.pointer_type(staging_ptr.dtype.element_ty))

        # Destination slot per token (within the layer's flat
        # [2, num_blocks * block_size, hidden_dim] view).
        # Cast to int64 BEFORE the stride multiply: for large
        # gpt-oss-120b-class caches (262K+ blocks × 16 × 512 hidden_dim)
        # the per-layer K-V offset crosses 2^31 and overflows int32.
        # Triton's default scalar type for loaded i32 indices is i32;
        # promoting once here makes all downstream arithmetic 64-bit.
        slot_indices = tl.load(slot_mapping_ptr + token_offsets, mask=token_mask, other=0).to(
            tl.int64
        )
        # Within the layer tensor: K-base + slot_idx*hidden_dim, or
        # V-base + slot_idx*hidden_dim. K-base = 0; V-base = layer_k_stride.
        #
        # Two physical layouts for regular (NUM_KV_CHANNELS=2) attention:
        #   OUTERMOST [2, num_blocks, block_size, H] (KV_INTERLEAVED=0): K and
        #     V are two separate contiguous [num_blocks*block_size, H] regions,
        #     so V-base = layer_k_stride = num_blocks*block_size*H and the row
        #     offset is simply slot*H. This is the base formula below.
        #   INTERLEAVED [num_blocks, 2, block_size, H] (KV_INTERLEAVED=1,
        #     vLLM 0.23 FullAttention): each block stores its K then its V
        #     contiguously, so a block occupies 2*block_size*H elements and
        #     V-base = layer_k_stride = block_size*H. The base formula lands a
        #     block b at b*block_size*H, but its true start is b*(2*block_size*H)
        #     — short by b*block_size*H (the intervening K OR V half of every
        #     earlier block). Add that per-block gap: (slot // block_size) is b.
        kv_base = tl.cast(kv_idx, tl.int64) * tl.cast(layer_k_stride, tl.int64)
        layer_token_base = kv_base + slot_indices * tl.cast(layer_slot_stride, tl.int64)
        if KV_INTERLEAVED == 1:
            block_idx = slot_indices // tl.cast(block_size, tl.int64)
            layer_token_base += (
                block_idx * tl.cast(block_size, tl.int64) * tl.cast(hidden_dim, tl.int64)
            )

        # Source row in staging: offset =
        #   kv_idx * (num_layers * chunk_tokens * hidden_dim)
        # + layer_idx * (chunk_tokens * hidden_dim)
        # + token_offset * hidden_dim
        # int64 arithmetic — for very wide chunks (Kimi MLA ~34 MiB)
        # the per-row offset can cross 2^31.
        token_offsets_i64 = token_offsets.to(tl.int64)
        staging_row_base = (
            tl.cast(kv_idx, tl.int64)
            * tl.cast(num_layers, tl.int64)
            * tl.cast(chunk_tokens, tl.int64)
            * tl.cast(hidden_dim, tl.int64)
            + tl.cast(layer_idx, tl.int64)
            * tl.cast(chunk_tokens, tl.int64)
            * tl.cast(hidden_dim, tl.int64)
            + token_offsets_i64 * tl.cast(hidden_dim, tl.int64)
        )

        # Inner loop over hidden_dim — Triton unrolls when hidden_dim
        # is small (≤ 256 elements); for wider rows it tiles. We process
        # 64 elements at a time so the kernel works for any
        # hidden_dim ≥ 64; smaller hidden_dim is also supported via the
        # mask on the tail iteration.
        HIDDEN_TILE: tl.constexpr = 64
        num_hidden_iters: tl.constexpr = (hidden_dim + HIDDEN_TILE - 1) // HIDDEN_TILE
        for hi in tl.static_range(num_hidden_iters):
            hidden_offs = hi * HIDDEN_TILE + tl.arange(0, HIDDEN_TILE)
            hidden_mask = hidden_offs < hidden_dim
            # 2D mask: token × hidden
            mask = token_mask[:, None] & hidden_mask[None, :]
            staging_offs = staging_row_base[:, None] + hidden_offs[None, :]
            layer_offs = layer_token_base[:, None] + hidden_offs[None, :]
            if DIRECTION == 0:
                # LOAD: staging → layer
                data = tl.load(staging_ptr + staging_offs, mask=mask, other=0)
                tl.store(layer_ptr + layer_offs, data, mask=mask)
            else:
                # SAVE: layer → staging
                data = tl.load(layer_ptr + layer_offs, mask=mask, other=0)
                tl.store(staging_ptr + staging_offs, data, mask=mask)


# ----------------------------------------------------------------------
# Public API — both directions, with shape validation + fallback path
# ----------------------------------------------------------------------


def _compute_slot_mapping(
    per_page_block_ids: Sequence[Sequence[int]],
    layer_to_group: Sequence[int],
    layer_idx: int,
    block_size: int,
) -> list[int]:
    """For layer `layer_idx`, return a per-token slot-mapping array
    of length ``chunk_tokens``. ``slot_mapping[t] = block_id_of_page *
    block_size + (t % block_size)`` where ``block_id_of_page`` comes
    from this layer's cache_group.

    Lives in Python because building it is O(chunk_tokens) and dwarfs
    the kernel time only at tiny chunk sizes; baked into a tensor
    once per chunk on the worker before the kernel call.
    """
    group_id = layer_to_group[layer_idx] if layer_idx < len(layer_to_group) else 0
    flat: list[int] = []
    for page_id_tuple in per_page_block_ids:
        # Defensive: if the page tuple is shorter than expected group
        # count, fall back to index 0.
        bid = page_id_tuple[group_id] if group_id < len(page_id_tuple) else page_id_tuple[0]
        base = int(bid) * int(block_size)
        for off in range(block_size):
            flat.append(base + off)
    return flat


def kv_chunk_scatter(
    staging: torch.Tensor,
    layer_tensors: list[torch.Tensor],
    per_page_block_ids: Sequence[Sequence[int]],
    layer_to_group: Sequence[int],
    block_size: int,
    use_triton: bool = True,
    sync: bool = False,
) -> None:
    """LOAD path: scatter ``staging`` into the per-layer paged KV
    caches. Shapes:
      - ``staging``: ``[2, num_layers, chunk_tokens, hidden_dim]`` —
        contiguous device tensor (dtype matches layer dtype).
      - ``layer_tensors[i]``: ``[2, num_blocks, block_size, num_kv_heads,
        head_dim]`` where ``num_kv_heads × head_dim == hidden_dim``.
      - ``per_page_block_ids[page_idx][group_id]`` = physical block_id
        in that layer's cache group.
      - ``layer_to_group[layer_idx]`` = which group each layer belongs to.

    Falls back to a pure-Python ``slot.copy_`` loop if Triton is
    unavailable OR if ``use_triton=False`` (set by unit tests as a
    correctness oracle).

    staging shape: [num_kv_channels, num_layers, chunk_tokens, hidden_dim]
    where num_kv_channels = 2 (K+V split, regular attention) or 1 (MLA).
    """
    num_kv_channels, num_layers, chunk_tokens, hidden_dim = staging.shape
    assert num_kv_channels in (1, 2)
    assert len(layer_tensors) == num_layers
    assert chunk_tokens % block_size == 0
    if not use_triton or not _TRITON_AVAILABLE or staging.device.type == "cpu":
        return _kv_chunk_scatter_python(
            staging, layer_tensors, per_page_block_ids, layer_to_group, block_size
        )
    return _kv_chunk_transfer_triton(
        staging,
        layer_tensors,
        per_page_block_ids,
        layer_to_group,
        block_size,
        direction=0,
        sync=sync,
    )


def kv_chunk_gather(
    staging: torch.Tensor,
    layer_tensors: list[torch.Tensor],
    per_page_block_ids: Sequence[Sequence[int]],
    layer_to_group: Sequence[int],
    block_size: int,
    use_triton: bool = True,
    sync: bool = False,
) -> None:
    """SAVE path: gather from per-layer paged KV caches into
    ``staging``. Same shapes as :func:`kv_chunk_scatter`; reverses
    direction so caller can serialize ``staging`` to disk."""
    num_kv_channels, num_layers, chunk_tokens, hidden_dim = staging.shape
    assert num_kv_channels in (1, 2)
    assert len(layer_tensors) == num_layers
    assert chunk_tokens % block_size == 0
    if not use_triton or not _TRITON_AVAILABLE or staging.device.type == "cpu":
        return _kv_chunk_gather_python(
            staging, layer_tensors, per_page_block_ids, layer_to_group, block_size
        )
    return _kv_chunk_transfer_triton(
        staging,
        layer_tensors,
        per_page_block_ids,
        layer_to_group,
        block_size,
        direction=1,
        sync=sync,
    )


def _kv_chunk_transfer_triton(
    staging: torch.Tensor,
    layer_tensors: list[torch.Tensor],
    per_page_block_ids: Sequence[Sequence[int]],
    layer_to_group: Sequence[int],
    block_size: int,
    direction: int,
    sync: bool = False,
) -> None:
    """Inner Triton launcher — shared by scatter (direction=0) and
    gather (direction=1). Builds the layer_ptrs + slot_mapping tensors
    on-the-fly; for a hot-path scheduler call we could pool these
    pre-allocated, but at the v2.0 stage we measure first."""
    num_kv_channels, num_layers, chunk_tokens, hidden_dim = staging.shape
    if num_kv_channels not in (1, 2):
        raise ValueError(
            f"staging leading dim must be 1 (MLA) or 2 (regular); got {num_kv_channels}"
        )
    device = staging.device

    # Per-layer base pointers — one int64 per layer.
    layer_ptrs = torch.tensor(
        [int(t.data_ptr()) for t in layer_tensors],
        dtype=torch.int64,
        device=device,
    )

    # Per-layer strides. Three layouts we support:
    #   regular OUTERMOST (num_kv_channels=2): [2, num_blocks, block_size, num_kv_heads, head_dim]
    #     K/V are two separate contiguous regions → K-base→V-base distance =
    #     num_blocks * block_size * hidden_dim; per-block gap term unused.
    #   regular INTERLEAVED (num_kv_channels=2): [num_blocks, 2, block_size, num_kv_heads, head_dim]
    #     (vLLM 0.23 FullAttention) K then V contiguous WITHIN each block →
    #     K-base→V-base distance = block_size * hidden_dim, and the kernel adds
    #     a per-block gap of block_size*hidden_dim (see KV_INTERLEAVED there).
    #   MLA (num_kv_channels=1): [num_blocks, block_size, hidden_dim]
    #     no K/V split — layer_k_stride is unused inside the kernel; pass 0.
    sample = layer_tensors[0]
    kv_interleaved = 0
    if num_kv_channels == 2:
        # Distinguish the two regular layouts by which axis holds the size-2
        # K/V split. shape[0]==2 → outermost (num_blocks never 2 in practice);
        # else shape[1]==2 → interleaved (the 0.23 FullAttention layout).
        if sample.shape[0] == 2:
            num_blocks = sample.shape[1]
            layer_k_stride = num_blocks * block_size * hidden_dim
        elif sample.dim() >= 4 and sample.shape[1] == 2:
            kv_interleaved = 1
            num_blocks = sample.shape[0]
            layer_k_stride = block_size * hidden_dim
        else:
            raise ValueError(
                f"regular-attention layer tensor has neither a leading nor a "
                f"dim-1 size-2 K/V axis: shape={tuple(sample.shape)}"
            )
    else:
        num_blocks = sample.shape[0]
        layer_k_stride = 0
    layer_slot_stride = hidden_dim

    # Each layer's slot_mapping is the SAME if all layers share one
    # cache group; for HMA it differs per group. Build one slot_mapping
    # per group, then for each layer pick its group's mapping at
    # kernel-launch time. To keep the kernel simple, we run one kernel
    # invocation per (group), passing only that group's layer indices
    # — for v2.0 a slot mapping tensor per layer is fine; future
    # optimization can collapse.

    # Find unique groups + their layer indices.
    group_to_layers: dict[int, list[int]] = {}
    for layer_idx, gid in enumerate(layer_to_group[:num_layers]):
        group_to_layers.setdefault(int(gid), []).append(layer_idx)
    if not group_to_layers:
        # No HMA info (single-group default): all layers in group 0.
        group_to_layers = {0: list(range(num_layers))}

    TOKEN_BLOCK = 64
    # Keep every device tensor handed to an async kernel launch alive
    # until AFTER we synchronize (when sync=True). The Triton kernel runs
    # async on the current stream; if `slot_mapping` / `group_layer_ptrs`
    # are freed when this function returns (before the kernel actually
    # runs), the caching allocator can hand their blocks to another thread
    # — fatal in the multi-threaded parallel/async load path, where a
    # reused block corrupts the in-flight kernel's slot indices and the
    # scatter writes to a wild address ("Memory access fault / write to a
    # read-only page"). In single-threaded modes stream-ordered frees make
    # this safe, but the parallel loaders pass sync=True. See
    # _start_load_kv_parallel.
    _keepalive: list = [layer_ptrs]
    for gid, layer_indices in group_to_layers.items():
        # Per-page block ids for this group — ONE int per page (cheap
        # Python work, O(n_pages)). The per-token expansion (× block_size)
        # is done on the GPU below, NOT in a Python loop: an
        # O(n_pages*block_size) Python loop here is held under the GIL and
        # serializes the parallel-load worker threads, which is what
        # collapsed 16-worker throughput back toward single-stream. Keeping
        # the GIL-held work O(n_pages) lets the workers actually fan out.
        bids = [
            int(page_id_tuple[gid] if gid < len(page_id_tuple) else page_id_tuple[0])
            for page_id_tuple in per_page_block_ids
        ]
        # Bounds guard (CPU-side, no GPU sync): the largest slot is
        # max_bid*block_size + (block_size-1). A slot >= the layer's row
        # capacity makes the kernel read/write past the paged-KV buffer →
        # silent "Memory access fault" GPU crash. Surface the offending
        # block_id and raise; every caller wraps the scatter/gather in
        # try/except and degrades to a cache miss (re-prefill), so one bad
        # chunk can't crash the worker. Per-channel row capacity: regular
        # attention packs 2 channels (K,V), MLA 1. Valid slot < cap_rows.
        _cap_rows = int(
            layer_tensors[layer_indices[0]].numel() // max(num_kv_channels * hidden_dim, 1)
        )
        _max_slot = (max(bids) * block_size + block_size - 1) if bids else -1
        if _max_slot >= _cap_rows:
            _bad = sorted({b for b in bids if b * block_size >= _cap_rows})
            logger.error(
                "kv_chunk OOB GUARD dir=%d gid=%d block_size=%d hidden_dim=%d "
                "cap_rows=%d max_slot=%d n_pages=%d chunk_tokens=%d bad_block_ids=%s",
                direction,
                gid,
                block_size,
                hidden_dim,
                _cap_rows,
                _max_slot,
                len(per_page_block_ids),
                chunk_tokens,
                _bad[:16],
            )
            raise ValueError(
                f"kv_chunk slot {_max_slot} >= layer capacity {_cap_rows} "
                f"(block_size={block_size}, bad_block_ids={_bad[:8]})"
            )
        # Vectorized GPU build: slot[page, off] = bid*block_size + off,
        # flattened to [n_pages*block_size]. One small H2D of the block ids
        # + a GPU broadcast — instead of materializing the full slot list
        # in Python. int64 math (avoids overflow on huge caches) → int32
        # (row indices fit; the kernel re-promotes to int64 for byte math).
        bids_t = torch.as_tensor(bids, dtype=torch.int64, device=device)
        slot_mapping = (
            (
                bids_t[:, None] * block_size
                + torch.arange(block_size, device=device, dtype=torch.int64)
            )
            .reshape(-1)
            .to(torch.int32)
        )

        # Filter layer_ptrs to only this group's layers — Triton kernel
        # treats layer_idx as a contiguous index, so we re-pack the
        # per-group layer ptrs in their original order.
        group_layer_ptrs = layer_ptrs[layer_indices]
        group_num_layers = len(layer_indices)

        # For this group, we need to launch the kernel with a staging
        # view restricted to these layers. Slice along dim 1
        # (num_layers axis). The staging tensor is contiguous, so a
        # slice is a view (no copy) but with a non-default stride —
        # to keep the kernel address math correct (which assumes
        # contiguous), we materialize a contiguous copy ONLY if the
        # group covers a non-contiguous layer subset. For the common
        # case of one group covering all layers, the .narrow returns
        # the original tensor.
        if sorted(layer_indices) == layer_indices and layer_indices == list(
            range(layer_indices[0], layer_indices[0] + len(layer_indices))
        ):
            # Contiguous slice — view is fine.
            staging_group = staging.narrow(1, layer_indices[0], group_num_layers)
            # The kernel still uses `num_layers = group_num_layers`
            # for its stride math because we pass the view's shape.
        else:
            # Non-contiguous group — gather to a temp. Rare; for HMA
            # models with alternating groups, this triggers.
            staging_group = staging.index_select(
                1, torch.tensor(layer_indices, device=device, dtype=torch.long)
            ).contiguous()

        _keepalive.append(slot_mapping)
        _keepalive.append(group_layer_ptrs)
        _keepalive.append(staging_group)
        grid = (num_kv_channels * group_num_layers, (chunk_tokens + TOKEN_BLOCK - 1) // TOKEN_BLOCK)
        _kv_chunk_transfer_kernel[grid](
            staging_group,
            group_layer_ptrs,
            slot_mapping,
            layer_k_stride=layer_k_stride,
            layer_slot_stride=layer_slot_stride,
            num_layers=group_num_layers,
            chunk_tokens=chunk_tokens,
            hidden_dim=hidden_dim,
            block_size=block_size,
            NUM_KV_CHANNELS=num_kv_channels,
            KV_INTERLEAVED=kv_interleaved,
            DIRECTION=direction,
            TOKEN_BLOCK=TOKEN_BLOCK,
        )

        # If we materialized a temp staging_group for save direction,
        # write the gathered bytes back into the original staging view.
        if direction == 1 and staging_group.data_ptr() != staging.data_ptr():
            # Save direction wrote into staging_group; copy back into
            # the right slices of the original staging tensor.
            for dst_idx, src_idx in enumerate(layer_indices):
                staging[:, src_idx].copy_(staging_group[:, dst_idx])

    # Block until the async kernel(s) finish while their arg tensors are
    # STILL referenced (via _keepalive), so freeing them on return can't
    # race a concurrent allocation. Required for the multi-threaded
    # parallel/async load path; cheap for single-chunk callers.
    if sync:
        torch.cuda.current_stream(device).synchronize()
    # _keepalive is dropped only here, after the sync.
    del _keepalive


# ----------------------------------------------------------------------
# Python-loop fallback (correctness oracle + no-Triton fallback)
# ----------------------------------------------------------------------


def _kv_chunk_scatter_python(
    staging: torch.Tensor,
    layer_tensors: list[torch.Tensor],
    per_page_block_ids: Sequence[Sequence[int]],
    layer_to_group: Sequence[int],
    block_size: int,
) -> None:
    """LOAD via slot.copy_ per (layer, page). Slow (re-introduces the
    per-call overhead chunk fusion exists to avoid) but correct.

    Handles three layouts:
      - NUM_KV_CHANNELS=2 outermost:   staging [2, L, T, H], layer [2, B, P, H]  → layer[:, bid]
      - NUM_KV_CHANNELS=2 interleaved: staging [2, L, T, H], layer [B, 2, P, H]  → layer[bid]  (vLLM 0.23)
      - NUM_KV_CHANNELS=1 (MLA):       staging [1, L, T, H], layer [B, P, H]
    """
    num_kv_channels, num_layers, chunk_tokens, hidden_dim = staging.shape
    pages_per_chunk = chunk_tokens // block_size
    for layer_idx in range(num_layers):
        gid = layer_to_group[layer_idx] if layer_idx < len(layer_to_group) else 0
        layer_tensor = layer_tensors[layer_idx]
        for page_idx in range(pages_per_chunk):
            page_ids = per_page_block_ids[page_idx]
            bid = page_ids[gid] if gid < len(page_ids) else page_ids[0]
            src = staging[
                :,
                layer_idx,
                page_idx * block_size : (page_idx + 1) * block_size,
                :,
            ]
            if num_kv_channels == 2:
                # dst is [2, block_size, ...]: layer[:, bid] for the outermost
                # layout, layer[bid] for the 0.23 interleaved [B, 2, P, H].
                dst = layer_tensor[bid] if layer_tensor.shape[0] != 2 else layer_tensor[:, bid]
                dst.copy_(src.view(dst.shape))
            else:
                # MLA: treat the layer as a flat [total_rows, hidden]
                # buffer and write this block's `block_size` rows at
                # [bid*block_size : ...]. Mirrors the Triton kernel's
                # slot = bid*block_size + offset addressing, so it is
                # correct for BOTH the [num_blocks, block_size, hidden]
                # layout AND the AITER [num_blocks*block_size, 1, hidden]
                # layout (middle dim 1, logical block_size > 1). `.view`
                # (not reshape) keeps the write aliased to the real tensor.
                flat = layer_tensor.view(-1, hidden_dim)
                base = bid * block_size
                flat[base : base + block_size].copy_(src.squeeze(0).reshape(block_size, hidden_dim))


def _kv_chunk_gather_python(
    staging: torch.Tensor,
    layer_tensors: list[torch.Tensor],
    per_page_block_ids: Sequence[Sequence[int]],
    layer_to_group: Sequence[int],
    block_size: int,
) -> None:
    """SAVE via slot.copy_ per (layer, page). Symmetric to scatter."""
    num_kv_channels, num_layers, chunk_tokens, hidden_dim = staging.shape
    pages_per_chunk = chunk_tokens // block_size
    for layer_idx in range(num_layers):
        gid = layer_to_group[layer_idx] if layer_idx < len(layer_to_group) else 0
        layer_tensor = layer_tensors[layer_idx]
        for page_idx in range(pages_per_chunk):
            page_ids = per_page_block_ids[page_idx]
            bid = page_ids[gid] if gid < len(page_ids) else page_ids[0]
            dst = staging[
                :,
                layer_idx,
                page_idx * block_size : (page_idx + 1) * block_size,
                :,
            ]
            if num_kv_channels == 2:
                # src is [2, block_size, ...]: layer[:, bid] for the outermost
                # layout, layer[bid] for the 0.23 interleaved [B, 2, P, H].
                src = layer_tensor[bid] if layer_tensor.shape[0] != 2 else layer_tensor[:, bid]
                dst.copy_(src.view(dst.shape))
            else:
                # MLA: gather this block's `block_size` rows from the flat
                # [total_rows, hidden] view at [bid*block_size : ...].
                # Layout-agnostic, symmetric to the scatter above.
                flat = layer_tensor.view(-1, hidden_dim)
                base = bid * block_size
                dst.squeeze(0).copy_(flat[base : base + block_size].reshape(block_size, hidden_dim))
