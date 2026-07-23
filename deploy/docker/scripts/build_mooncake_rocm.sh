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
#   RDMA via host-libionic injection); 1 = main @ pinned ref + B-group C++ patches
#   (B.2 hip-transport gate + B.3 auto-chunk MR) for DSv4 cross-node RDMA.
#
# GPU MR registration path (independent of the mode above), via MOONCAKE_HIP_DMABUF
#   (default 0): 0 = bare ibv_reg_mr (needs the legacy ib_peer_mem kernel module,
#   the historical default; do NOT use at high util — HIP-209 note below); 1 =
#   dma-buf GPUDirect (B.1: hsa_amd_portable_export_dmabuf + ibv_reg_dmabuf_mr),
#   REQUIRED on dma-buf-only RoCE fabrics that have NO ib_peer_mem (e.g. the crusoe
#   amd-spur cluster), where bare ibv_reg_mr can't pin VRAM and OOMs (>100 GiB host
#   pin). B.1 was dropped in #154 because dma-buf at high util (>=0.7) exhausts a
#   KFD resource -> every later hipModuleLoad fails HIP-209; it is re-added here as
#   an opt-in so low-util / dma-buf-only fabrics work while ib_peer_mem + high-util
#   production keeps bare ibv_reg_mr. Only takes effect in mode 1 (MOONCAKE_DMABUF=1),
#   where the B-group patches are applied.
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
export MOONCAKE_HIP_DMABUF  # read by apply_mooncake_cpp_patches.sh (B.1 gate)
MOONCAKE_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MC_ROOT="${MC_ROOT:-/opt/mooncake/Mooncake}"
MC_CPP_PATCH_DIR="${MC_CPP_PATCH_DIR:-$SCRIPT_DIR/patches/mooncake_cpp}"

if [ "$MOONCAKE_DMABUF" = "1" ]; then
    MOONCAKE_GIT_REF="${MOONCAKE_GIT_REF:-747003c058015c4077a266e7ccd7549bbc9baede}"
    if [ "$MOONCAKE_HIP_DMABUF" = "1" ]; then
        MODE_DESC="main + B-group C++ patches + B.1 dma-buf GPUDirect (ibv_reg_dmabuf_mr)"
    else
        MODE_DESC="main + B-group C++ patches (bare ibv_reg_mr, needs ib_peer_mem)"
    fi
    # main defaults RUST store ON; turn it off and pin pybind11. The B.1 patch
    # (applied when MOONCAKE_HIP_DMABUF=1) turns on the dma-buf verb for the
    # rdma_transport object lib; when off, GPU regs use bare ibv_reg_mr.
    # pybind11_DIR: auto-detect from the ACTIVE interpreter so this works across
    # images with different Python layouts (vLLM py3.12 /usr/local; sglang py3.10
    # /opt/venv). Fall back to the historical py3.12 path if pybind11 can't say.
    PYBIND11_CMAKE_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())' 2>/dev/null \
        || echo /usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11)"
    EXTRA_CMAKE=(-DWITH_STORE_RUST=OFF -Dpybind11_DIR="$PYBIND11_CMAKE_DIR")
else
    MOONCAKE_GIT_REF="${MOONCAKE_GIT_REF:-v0.3.7.post2}"
    MODE_DESC="release, no dma-buf (host-libionic injection)"
    EXTRA_CMAKE=()
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
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/usr/local/lib/python3.12/dist-packages/pybind11/share/cmake/pybind11:${CMAKE_PREFIX_PATH:-}"
ENGINE_SO_GLOB="build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so"
if ! ls $ENGINE_SO_GLOB >/dev/null 2>&1; then
    echo "=== cmake configure (USE_HIP=ON ${EXTRA_CMAKE[*]:-}) ==="
    rm -rf build && mkdir build && cd build
    cmake .. -DUSE_HIP=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
        -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF "${EXTRA_CMAKE[@]}" \
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

# ---- verify ----------------------------------------------------------------
SO="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
echo "installed: $SO"
# When B.1 (dma-buf) is enabled, assert the dmabuf verb + HSA export symbol were
# actually compiled into the engine .so — else the CMake define silently didn't
# reach rdma_transport and we'd be back to the OOM-on-dma-buf-fabric bare path.
if [ "$MOONCAKE_HIP_DMABUF" = "1" ]; then
    # Read the .dynsym (undefined) entries once; the dma-buf branch pulls in
    # ibv_reg_dmabuf_mr + hsa_amd_portable_export_dmabuf as undefined dynamic
    # symbols. Use plain `nm -D` (portable across the engine images' binutils;
    # `nm -DC`/`nm -C`/`strings` behave inconsistently on some ROCm bases and
    # gave false negatives) and grep a captured copy so `set -o pipefail`
    # can't trip on grep's exit status.
    _dynsyms="$(nm -D "$SO" 2>/dev/null || true)"
    if printf '%s\n' "$_dynsyms" | grep -qE 'ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf'; then
        echo "MC_DMABUF_VERIFY OK (dma-buf GPUDirect symbols present)"
    else
        echo "ERROR: MOONCAKE_HIP_DMABUF=1 but dma-buf symbols absent from $SO" >&2
        echo "       -> B.1 CMake patch did not reach rdma_transport; check build log." >&2
        exit 1
    fi
fi
python3 -c "from mooncake.engine import TransferEngine; print('MOONCAKE IMPORT OK')"
echo "MC_BUILD_DONE"
