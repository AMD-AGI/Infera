#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Gate Mooncake's HIP IPC transport behind MC_ENABLE_HIP_TRANSPORT. Run once per
# node, BEFORE the engine legs — this is the containerised equivalent of what
# deploy/docker/Dockerfile.sglang does at image build time, for clusters where we
# only get a running container and cannot rebuild the image.
#
#   bash patch_mooncake_hip.sh            # patch, rebuild, install, verify
#   bash patch_mooncake_hip.sh --status   # report only
#   bash patch_mooncake_hip.sh --restore  # put the stock engine.so back
#
# WHY: the bundled Mooncake (upstream #2682, commit 01d1eb2a) installs a HIP IPC
# transport unconditionally and selectTransport prefers it over RDMA. In 1P1D the
# two legs are on different nodes, so the KV hand-off picks a transport that
# cannot possibly work and every request dies with
#     hip_transport.cpp:70] hipIpcOpenMemHandle failed (Error code: 17 - invalid
#     device pointer)
#     Prefill transfer failed ... Failed to send kv chunk
# MC_USE_HIP_IPC=0 does NOT help: it tunes the transport, it does not stop it from
# being installed and preferred. Only the source gate does.
#
# The build is incremental (one translation unit) and takes ~30 s. The resulting
# .so is cached on shared storage keyed by the stock .so's hash, so the second
# node installs it instead of rebuilding — and refuses to if its stock .so differs,
# which would mean the two nodes are not running the same image.
set -euo pipefail

MC_ROOT="${MC_ROOT:-/sgl-workspace/Mooncake}"
REPO="${REPO:-$(cd "$(dirname "$0")/../.." && pwd)}"
PATCH="${PATCH:-$REPO/deploy/docker/patches/mooncake_cpp/transfer_engine_impl.diff}"
CACHE_DIR="${CACHE_DIR:-/wekafs/llying/tmp/mooncake-hipgate}"
JOBS="${JOBS:-$(nproc)}"
GATE="MC_ENABLE_HIP_TRANSPORT"

DEST="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
BACKUP="${DEST}.stock"

# `strings | grep -q` closes the pipe early and SIGPIPEs strings under pipefail,
# which reads as "not patched" no matter what. Count into a variable instead.
has_gate() { [[ "$(strings "$1" 2>/dev/null | grep -c "$GATE" || true)" != "0" ]]; }

status() {
    printf 'engine.so : %s\n' "$DEST"
    printf 'gate      : %s\n' "$(has_gate "$DEST" && echo "present (HIP IPC is opt-in)" || echo "ABSENT (HIP IPC forced on)")"
    printf 'backup    : %s\n' "$([[ -f "$BACKUP" ]] && echo "$BACKUP" || echo "none")"
    printf 'source    : %s @ %s\n' "$MC_ROOT" "$(git -C "$MC_ROOT" rev-parse --short HEAD 2>/dev/null || echo '?')"
}

case "${1:-}" in
    --status) status; exit 0 ;;
    --restore)
        [[ -f "$BACKUP" ]] || { echo "[mc-hip] no backup at $BACKUP" >&2; exit 1; }
        cp "$BACKUP" "$DEST"
        echo "[mc-hip] restored stock engine.so"; status; exit 0 ;;
    "") ;;
    *) echo "usage: bash $0 [--status|--restore]" >&2; exit 2 ;;
esac

if has_gate "$DEST"; then
    echo "[mc-hip] already gated — nothing to do"
    status
    exit 0
fi

[[ -f "$BACKUP" ]] || cp "$DEST" "$BACKUP"
STOCK_SHA="$(sha256sum "$BACKUP" | cut -c1-16)"
CACHED="$CACHE_DIR/engine-hipgate-${STOCK_SHA}.so"
echo "[mc-hip] stock engine.so ${STOCK_SHA}, cache ${CACHED}"

if [[ -f "$CACHED" ]]; then
    echo "[mc-hip] installing cached build (another node already built it)"
    cp "$CACHED" "$DEST"
else
    [[ -f "$MC_ROOT/mooncake-transfer-engine/src/transfer_engine_impl.cpp" ]] \
        || { echo "[mc-hip] $MC_ROOT is not a Mooncake source tree" >&2; exit 1; }
    [[ -f "$PATCH" ]] || { echo "[mc-hip] patch not found: $PATCH" >&2; exit 1; }

    if git -C "$MC_ROOT" apply --reverse --check "$PATCH" 2>/dev/null; then
        echo "[mc-hip] source already patched"
    else
        git -C "$MC_ROOT" apply "$PATCH"
        echo "[mc-hip] source patched"
    fi

    # Reuse the image's configure step when it survived; only the one edited
    # translation unit recompiles. Pin the arch so the build does not depend on
    # ROCm's amdgpu-arch probe finding a free GPU.
    ARCH="${GPU_ARCH:-$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1 || true)}"
    ARCH="${ARCH:-gfx942}"
    export PYTORCH_ROCM_ARCH="$ARCH" GPU_ARCHS="$ARCH" AMDGPU_TARGETS="$ARCH" HIP_ARCHITECTURES="$ARCH"
    if [[ ! -f "$MC_ROOT/build/CMakeCache.txt" ]]; then
        echo "[mc-hip] no build tree — configuring (arch=$ARCH)"
        mkdir -p "$MC_ROOT/build"
        ( cd "$MC_ROOT/build" && cmake .. -DUSE_HIP=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
            -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_STORE_RUST=OFF \
            "-DCMAKE_HIP_ARCHITECTURES=$ARCH" >/dev/null )
    fi
    # "engine" is the pybind module target; the plain .so filename is not a target.
    echo "[mc-hip] building engine.so (arch=$ARCH, -j$JOBS)"
    ( cd "$MC_ROOT/build" && make engine -j"$JOBS" >/dev/null )

    BUILT="$MC_ROOT/build/mooncake-integration/$(basename "$DEST")"
    [[ -f "$BUILT" ]] || { echo "[mc-hip] build produced no $BUILT" >&2; exit 1; }
    has_gate "$BUILT" || { echo "[mc-hip] rebuilt .so has no gate — patch did not compile in" >&2; exit 1; }
    cp "$BUILT" "$DEST"
    mkdir -p "$CACHE_DIR" && cp "$BUILT" "$CACHED"
    echo "[mc-hip] cached -> $CACHED"
fi

has_gate "$DEST" || { echo "[mc-hip] installed .so still has no gate" >&2; exit 1; }
python3 -c "from mooncake.engine import TransferEngine" \
    || { echo "[mc-hip] installed .so does not import" >&2; exit 1; }
echo "[mc-hip] OK — HIP IPC is now opt-in; leave MC_ENABLE_HIP_TRANSPORT unset for cross-node PD"
status
