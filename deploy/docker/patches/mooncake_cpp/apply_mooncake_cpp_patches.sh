#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# WHAT: apply the DSv4 Mooncake C++ patches (B-group) to a Mooncake tree before
#   the cmake/ninja build — both required for cross-node GDR on AMD (MI355X+ionic).
#   B.1 CMakeLists: propagate USE_HIP_DMABUF+hsa-runtime64 to rdma_transport (else
#       the ibv_reg_dmabuf_mr branch compiles out -> bare ibv_reg_mr fails on ionic).
#   B.2 transfer_engine_impl: do NOT installTransport("hip") on ROCm by default
#       (else GPU bufs become intra-node hip IPC segments a cross-node peer can't
#       open — an IPC handle is host-local, so hipIpcOpenMemHandle on a peer NODE
#       fails with "invalid device context"/201). MC_ENABLE_HIP_TRANSPORT=1 opts
#       back in for single-node-only P2P; MC_DISABLE_HIP_TRANSPORT=1 vetoes even
#       that, so infera's rocm_rdma_env.py default is honoured rather than a no-op.
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

# B.1 (USE_HIP_DMABUF -> ibv_reg_dmabuf_mr) is OPT-IN. It was dropped in #154
# because dma-buf at high util exhausts a KFD resource (later hipModuleLoad ->
# HIP-209) where ib_peer_mem is available; but on a dma-buf-only fabric (no
# ib_peer_mem) it is the ONLY path that can register VRAM at all.
if [ "${MOONCAKE_HIP_DMABUF:-0}" = "1" ]; then
    echo "MOONCAKE_HIP_DMABUF=1 -> applying B.1 (dma-buf GPUDirect MR registration)"
    apply_one rdma_transport_dmabuf_cmake.diff
else
    echo "MOONCAKE_HIP_DMABUF=0 (default) -> B.1 not applied (bare ibv_reg_mr; needs ib_peer_mem)"
fi
apply_one transfer_engine_impl.diff
apply_one rdma_auto_chunk_mr_2017.diff

echo "MC_CPP_PATCHES_DONE"
