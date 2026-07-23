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

# B.1 (rdma_transport_dmabuf_cmake.diff, USE_HIP_DMABUF -> ibv_reg_dmabuf_mr) is
# OPT-IN, gated by MOONCAKE_HIP_DMABUF=1 (default 0 = off).
#
# History: B.1 was originally dropped because that dma-buf GPUDirect registration
# path exhausts a driver/KFD resource at high gpu_memory_utilization (>=0.7):
# registering the whole KV pool (~156 GiB/GPU at util 0.8) via ibv_reg_dmabuf_mr
# makes EVERY subsequent hipModuleLoad fail with HIP-209 "no kernel image is
# available for execution", crashing the decode engine at kernel_warmup / first
# inference. On fabrics that expose the legacy ib_peer_mem kernel module, bare
# ibv_reg_mr (VRAM RDMA via host-libionic injection) does NOT exhaust it and
# transfers KV correctly (validated util 0.8 P<->D on Kimi-K2.6 + DeepSeek-V4-Pro).
#
# BUT on a dma-buf-only RoCE fabric (no ib_peer_mem — e.g. the crusoe amd-spur
# cluster, which replaced ib_peer_mem with the kernel dma-buf path) bare
# ibv_reg_mr CANNOT pin VRAM: registering the KV pool degenerates to a huge
# host-memory pin and OOMs (needs >100 GiB of host RAM). There the dmabuf verb
# (hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr) is the ONLY working GPU
# registration path. So B.1 is re-added as an opt-in: set MOONCAKE_HIP_DMABUF=1
# for images that run on a dma-buf-only fabric, keep it 0 (default) on ib_peer_mem
# fabrics + high-util production to avoid the HIP-209 exhaustion. The dmabuf code
# in rdma_context.cpp still self-checks the kernel (CONFIG_PCI_P2PDMA /
# CONFIG_DMABUF_MOVE_NOTIFY) and falls back to ibv_reg_mr when unsupported, and
# MOONCAKE_DISABLE_HIP_DMABUF=1 is a runtime override.
if [ "${MOONCAKE_HIP_DMABUF:-0}" = "1" ]; then
    echo "MOONCAKE_HIP_DMABUF=1 -> applying B.1 (dma-buf GPUDirect MR registration)"
    apply_one rdma_transport_dmabuf_cmake.diff
else
    echo "MOONCAKE_HIP_DMABUF=0 (default) -> B.1 dma-buf path NOT applied (bare ibv_reg_mr; needs ib_peer_mem)"
fi
apply_one transfer_engine_impl.diff
apply_one rdma_auto_chunk_mr_2017.diff

echo "MC_CPP_PATCHES_DONE"
