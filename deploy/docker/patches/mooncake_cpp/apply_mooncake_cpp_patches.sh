#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# WHAT: apply Mooncake C++ patches before the cmake/ninja build. With no
#   arguments, apply B.2 below (vLLM and ATOM builds); the SGLang build passes
#   the two rdma_*.diff files instead.
#   B.2 transfer_engine_impl: do NOT installTransport("hip") on ROCm by default
#       (else GPU bufs become intra-node hip IPC segments a cross-node peer can't
#       open — an IPC handle is host-local, so hipIpcOpenMemHandle on a peer NODE
#       fails with "invalid device context"/201). MC_ENABLE_HIP_TRANSPORT=1 opts
#       back in for single-node-only P2P; MC_DISABLE_HIP_TRANSPORT=1 vetoes even
#       that, so infera's rocm_rdma_env.py default is honoured rather than a no-op.
#   rdma_dest_local_rail: MC_ENABLE_DEST_LOCAL_RAIL=1 (with the upstream
#       MC_ENABLE_DEST_DEVICE_AFFINITY) makes a WRITE to a peer GPU use the HCA
#       pinned for that GPU on both hosts. Off by default. Mooncake#4540.
#   rdma_ack_timeout: MC_IB_TIMEOUT sets the RC ACK timeout exponent
#       (4.096 us * 2^n); unset keeps the hardcoded 14, about 67 ms. Mooncake#4541.
#   (Gone, all upstream at the pinned ref: B.1 propagated USE_HIP_DMABUF to
#   rdma_transport, now done by src/CMakeLists.txt; B.3 auto-chunked MRs larger
#   than max_mr_size, now Mooncake#2644 along with the batch-registration error
#   propagation that used to swallow a failed KV registration.)
# VERSION: pinned to Mooncake main @ faae8dd4 (kvcache-ai/Mooncake). Plain
#   `git apply` fails loudly on ref drift rather than silently mis-patching.
# USAGE: MC_ROOT=/opt/mooncake/Mooncake bash apply_mooncake_cpp_patches.sh [DIFF...]
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

[ "$#" -gt 0 ] || set -- transfer_engine_impl.diff
for diff in "$@"; do
    apply_one "$diff"
done

echo "MC_CPP_PATCHES_DONE"
