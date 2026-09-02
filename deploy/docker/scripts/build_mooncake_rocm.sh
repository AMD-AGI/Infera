#!/bin/bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Build + install the Mooncake transfer engine from source with ROCm/HIP.
#
# Why this script exists:
#   The PyPI `mooncake-transfer-engine` wheel is CUDA-only (links libcudart.so.12)
#   and will not load on a ROCm host. vLLM's `MooncakeConnector` does
#   `from mooncake.engine import TransferEngine`, so for PD-disaggregation on AMD
#   (MI300X/MI355X + Pensando ionic) we must build Mooncake ourselves.
#
# Two build modes via MOONCAKE_DMABUF (default 0): 0 = release -DUSE_HIP=ON (VRAM
#   RDMA via host-libionic injection); 1 = main @ pinned ref + the B.2
#   hip-transport gate, for DSv4 cross-node RDMA.
#
# GPU MR registration path via MOONCAKE_HIP_DMABUF (default 0, mode 1 only):
#   0 = bare ibv_reg_mr, needs the legacy ib_peer_mem module; 1 = dma-buf
#   GPUDirect, the only path that registers VRAM where ib_peer_mem is absent.
#   It maps straight onto upstream's USE_HIP_DMABUF cmake option.
#
# Idempotent: reuses an existing build artifact if present.
#
# Environment overrides:
#   MOONCAKE_DMABUF      0 (release ref) | 1 (main ref + B-group patches)
#   MOONCAKE_HIP_DMABUF  0 (bare ibv_reg_mr) | 1 (dma-buf GPUDirect; mode 1 only)
#   MOONCAKE_GIT_REF     git tag/branch/commit (default depends on MOONCAKE_DMABUF)
#   MOONCAKE_REPO        git remote   (default https://github.com/kvcache-ai/Mooncake.git)
#   MC_ROOT              checkout dir (default /opt/mooncake/Mooncake)
#   MC_CPP_PATCH_DIR     B-group patch dir (mode 1 only)
#
# Usage (inside a ROCm container with hipcc/cmake/ninja/git):
#   bash deploy/docker/scripts/build_mooncake_rocm.sh                    # release ref
#   MOONCAKE_DMABUF=1 bash deploy/docker/scripts/build_mooncake_rocm.sh  # main + B-group
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export DEBIAN_FRONTEND=noninteractive

# ---- mode-dependent settings (all resolved up front) -----------------------
MOONCAKE_DMABUF="${MOONCAKE_DMABUF:-0}"
MOONCAKE_HIP_DMABUF="${MOONCAKE_HIP_DMABUF:-0}"
export MOONCAKE_HIP_DMABUF
MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MC_ROOT="${MC_ROOT:-/opt/mooncake/Mooncake}"
MC_CPP_PATCH_DIR="${MC_CPP_PATCH_DIR:-$SCRIPT_DIR/patches/mooncake_cpp}"

if [ "$MOONCAKE_DMABUF" = "1" ]; then
    MOONCAKE_GIT_REF="${MOONCAKE_GIT_REF:-faae8dd4a6309c3ecd47e0721a83b0250d686fa2}"
    # Upstream defaults USE_HIP_DMABUF ON, so pass it either way rather than
    # letting MOONCAKE_HIP_DMABUF=0 silently still compile the dma-buf path in.
    if [ "$MOONCAKE_HIP_DMABUF" = "1" ]; then
        MODE_DESC="main + B.2 gate + dma-buf GPUDirect (ibv_reg_dmabuf_mr)"
        DMABUF_CMAKE=(-DUSE_HIP_DMABUF=ON)
    else
        MODE_DESC="main + B.2 gate (bare ibv_reg_mr, needs ib_peer_mem)"
        DMABUF_CMAKE=(-DUSE_HIP_DMABUF=OFF)
    fi
    # main defaults RUST store ON; turn it off and pin pybind11. pybind11_DIR is
    # auto-detected from the ACTIVE interpreter, since the engine images lay
    # Python out differently (vLLM /usr/local, sglang + ATOM /opt/venv).
    PYBIND11_CMAKE_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null \
        || echo /usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11)"
    EXTRA_CMAKE=(-DWITH_STORE_RUST=OFF -Dpybind11_DIR="$PYBIND11_CMAKE_DIR")
else
    MOONCAKE_GIT_REF="${MOONCAKE_GIT_REF:-v0.3.7.post2}"
    MODE_DESC="release, no dma-buf (host-libionic injection)"
    EXTRA_CMAKE=()
    DMABUF_CMAKE=()
fi

echo "============================================"
echo "  Mooncake ROCm/HIP source build"
echo "  mode=${MODE_DESC}  ref=${MOONCAKE_GIT_REF}  root=${MC_ROOT}"
echo "============================================"

# Drop any pre-installed CUDA-only wheel so our source build takes over.
pip uninstall -y mooncake-transfer-engine >/dev/null 2>&1 || true

# ---- source tree (clone full + checkout ref: works for tags and commits) ---
if [ ! -d "$MC_ROOT/.git" ]; then
    mkdir -p "$(dirname "$MC_ROOT")"
    git clone "$MOONCAKE_REPO" "$MC_ROOT"
fi
cd "$MC_ROOT"
git fetch --all --tags >/dev/null 2>&1 || true
git checkout "$MOONCAKE_GIT_REF"
git submodule update --init --recursive
echo "mooncake HEAD: $(git rev-parse --short HEAD)"

# ---- apply B-group C++ patches (mode 1 / main only) ------------------------
if [ "$MOONCAKE_DMABUF" = "1" ]; then
    MC_ROOT="$MC_ROOT" bash "$MC_CPP_PATCH_DIR/apply_mooncake_cpp_patches.sh"
fi

# ---- system + third-party deps ---------------------------------------------
# dependencies.sh installs apt packages, yalantinglibs, glog/gflags, Go, and the
# git submodules Mooncake needs. -y for non-interactive container builds.
echo "=== dependencies.sh ==="
bash dependencies.sh -y 2>&1 | tail -15 || echo "dependencies.sh returned $? (continuing)"

# ---- configure + build -----------------------------------------------------
# USE_HIP=ON   : AMD/ROCm transport (links libamdhip64, not CUDA)
# USE_ETCD=OFF : we use vLLM's P2PHANDSHAKE bootstrap, not etcd metadata
# WITH_STORE=OFF + BUILD_UNIT_TESTS/EXAMPLES=OFF : transfer-engine only, trim build
# EXTRA_CMAKE  : mode 1 adds RUST-off / pybind11 pin (see above; no dma-buf)
# pybind11 on PREFIX_PATH so main's cmake finds the pip pybind11 (harmless otherwise).
# ROCm sits at /opt/rocm on the vllm-openai-rocm bases, but the ROCm 10.1 "ufb"
# bases ship it as a pip package (_rocm_sdk_devel) and point ROCM_PATH/ROCM_HOME
# at it instead, leaving no /opt/rocm. Same for pybind11: ask the interpreter
# rather than naming a dist-packages path that only exists under python3.12.
ROCM_ROOT="${ROCM_PATH:-${ROCM_HOME:-/opt/rocm}}"
PYBIND11_CMAKE="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null || true)"
export CMAKE_PREFIX_PATH="$ROCM_ROOT:$ROCM_ROOT/lib/cmake:/opt/rocm:/opt/rocm/lib/cmake:${PYBIND11_CMAKE:+$PYBIND11_CMAKE:}${CMAKE_PREFIX_PATH:-}"
ENGINE_SO_GLOB="build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so"
if ! ls $ENGINE_SO_GLOB >/dev/null 2>&1; then
    echo "=== cmake configure (USE_HIP=ON ${DMABUF_CMAKE[*]:-} ${EXTRA_CMAKE[*]:-}) ==="
    rm -rf build && mkdir build && cd build
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
        "${DMABUF_CMAKE[@]}" "${EXTRA_CMAKE[@]}" \
        -GNinja 2>&1 | tail -25
    echo "=== ninja build ==="
    ninja 2>&1 | tail -25
    cd "$MC_ROOT"
fi

# ---- assemble + install the python package ---------------------------------
ENGINE_SO="$(ls $ENGINE_SO_GLOB | head -1)"
if [ -z "${ENGINE_SO:-}" ] || [ ! -f "$ENGINE_SO" ]; then
    echo "ERROR: built engine .so not found ($ENGINE_SO_GLOB)" >&2
    exit 1
fi
cp "$ENGINE_SO" mooncake-wheel/mooncake/
pip install ./mooncake-wheel --no-deps --no-build-isolation 2>&1 | tail -10

# libasio.so (built alongside) must be on the loader path (the main build
# produces it; release build usually doesn't -> the glob is a no-op then).
ASIO_SO="$(ls build/mooncake-common/libasio.so 2>/dev/null | head -1 || true)"
if [ -n "${ASIO_SO:-}" ]; then
    cp "$ASIO_SO" /usr/local/lib/ && ldconfig
    echo "installed libasio.so"
fi

# ---- drop the Go toolchain dependencies.sh installed ------------------------
# dependencies.sh installs Go unconditionally, for the etcd metadata client and
# the Rust/Go store. We build with USE_ETCD=OFF and WITH_STORE=OFF, so nothing
# we ship links against it -- the engine is a Python extension over HIP/RDMA.
# Removed here, in the same RUN as the build, because a delete in a later layer
# leaves the files in the one below. Only /usr/local/go: bases that ship their
# own Go under $HOME/go own that copy.
if [ -d /usr/local/go ]; then
    rm -rf /usr/local/go
    echo "removed /usr/local/go (installed by dependencies.sh; USE_ETCD=OFF, WITH_STORE=OFF)"
fi

# ---- verify ----------------------------------------------------------------
SO="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
echo "installed: $SO"
# Assert the dmabuf verbs really compiled in — a CMake define that failed to
# reach rdma_transport would silently leave the bare ibv_reg_mr path. Capture
# `nm -D` into a var first so pipefail can't trip on grep's exit status.
if [ "$MOONCAKE_HIP_DMABUF" = "1" ]; then
    _dynsyms="$(nm -D "$SO" 2>/dev/null || true)"
    if printf '%s\n' "$_dynsyms" | grep -qE 'ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf'; then
        echo "MC_DMABUF_VERIFY OK (dma-buf GPUDirect symbols present)"
    else
        echo "ERROR: MOONCAKE_HIP_DMABUF=1 but dma-buf symbols absent from $SO" >&2
        echo "       -> USE_HIP_DMABUF did not reach rdma_transport; check build log." >&2
        exit 1
    fi
fi
# Assert the B.2 HIP-transport gate compiled in. This build does not enable
# upstream's ENABLE_MULTI_PROTOCOL locality routing, so without the gate the
# binary installs the HIP transport unconditionally and selectTransport
# prefers it over RDMA, so cross-node PD dies in KV transfer with
# "hipIpcOpenMemHandle failed (201 - invalid device context)" — an IPC handle is
# host-local, so a peer NODE can never open it. Both spellings must be present:
# MC_ENABLE_HIP_TRANSPORT (opt back in) and MC_DISABLE_HIP_TRANSPORT (hard veto,
# the spelling infera's rocm_rdma_env.py sets).
for _v in MC_ENABLE_HIP_TRANSPORT MC_DISABLE_HIP_TRANSPORT; do
    if ! strings "$SO" | grep -c "$_v" | grep -qv '^0$'; then
        echo "ERROR: $SO lacks $_v — the B.2 HIP-transport gate did not take." >&2
        echo "       Cross-node PD would break (hipIpcOpenMemHandle 201)." >&2
        exit 1
    fi
done
echo "MC_HIP_GATE_VERIFY OK (HIP transport OFF by default)"
# Assert the Go toolchain really is gone, so a future dependencies.sh that puts
# it somewhere else fails the build instead of silently shipping it again.
if [ -d /usr/local/go ]; then
    echo "ERROR: /usr/local/go still present after cleanup." >&2
    echo "       dependencies.sh moved the Go toolchain; update the rm above." >&2
    exit 1
fi
echo "MC_NO_GO_VERIFY OK (/usr/local/go absent)"
python3 -c "from mooncake.engine import TransferEngine; print('MOONCAKE IMPORT OK')"
echo "MC_BUILD_DONE"
