#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# WHAT: apply the DSv4 Mooncake C++ patches (B-group) to a Mooncake tree before
#   the cmake/ninja build — both required for cross-node GDR on AMD (MI355X+ionic).
#   B.1 CMakeLists: propagate USE_HIP_DMABUF+hsa-runtime64 to rdma_transport (else
#       the ibv_reg_dmabuf_mr branch compiles out -> bare ibv_reg_mr fails on ionic).
#   B.2 transfer_engine_impl: gate installTransport("hip") behind MC_ENABLE_HIP_TRANSPORT
#       (else GPU bufs become intra-node hip IPC segments a cross-node peer can't open).
#   B.3 rdma_auto_chunk_mr_2017: split buffers larger than the device max_mr_size
#       into <=max_mr_size MRs (one BufferDesc per chunk) instead of silently
#       truncating ibv_reg_mr — on ionic (max_mr_size ~2GiB) a truncated MR makes
#       remote ops past the boundary fail with IBV_WC_REM_ACCESS_ERR. Mooncake#2017
#       / upstream PR #2644 (not yet merged), so we carry it out-of-tree here.
# VERSION: diffs pinned to Mooncake main @ 747003c (kvcache-ai/Mooncake). Plain
#   `git apply` fails loudly on ref drift rather than silently mis-patching.
# USAGE: MC_ROOT=/opt/mooncake/Mooncake bash apply_mooncake_cpp_patches.sh
set -euo pipefail

MC_ROOT="${MC_ROOT:-/opt/mooncake/Mooncake}"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== B-group mooncake C++ patches -> $MC_ROOT ==="

apply_one() {
    local diff="$1" name
    name="$(basename "$diff")"
    # --reverse --check succeeds only if the patch is ALREADY applied -> idempotent skip.
    if git -C "$MC_ROOT" apply --reverse --check "$HERE/$diff" 2>/dev/null; then
        echo "[$name] already applied — skipping"
    elif git -C "$MC_ROOT" apply "$HERE/$diff"; then
        echo "[$name] applied"
    else
        echo "[$name] FAILED to apply — mooncake ref drifted from the pinned" \
             "commit? Update the diff for the new source." >&2
        exit 1
    fi
}

# B.1 (rdma_transport_CMakeLists.diff, USE_HIP_DMABUF -> ibv_reg_dmabuf_mr) REMOVED.
# That dma-buf GPUDirect registration path exhausts a driver/KFD resource at high
# gpu_memory_utilization (>=0.7): registering the whole KV pool (~156 GiB/GPU at
# util 0.8) via ibv_reg_dmabuf_mr makes EVERY subsequent hipModuleLoad fail with
# HIP-209 "no kernel image is available for execution", crashing the decode engine
# at kernel_warmup / first inference (_compute_slot_mapping_kernel, torch fill_,
# ...). The GPU BAR is 512 GiB (not the limit); reproduces on Kimi-K2.6 AND
# DeepSeek-V4-Pro. Bare ibv_reg_mr (VRAM RDMA via host-libionic injection) does NOT
# exhaust it and transfers KV correctly on ionic (validated util 0.8 P<->D, both
# models). The diff is dropped entirely — bare ibv_reg_mr is the only supported path.
apply_one transfer_engine_impl.diff
apply_one rdma_auto_chunk_mr_2017.diff

echo "MC_CPP_PATCHES_DONE"
