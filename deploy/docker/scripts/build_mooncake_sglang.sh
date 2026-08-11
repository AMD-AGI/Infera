#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Rebuild the SGLang image's Mooncake from one pinned upstream revision.
#
# That revision already contains the fixes Infera used to carry locally:
#   - HIP dma-buf is compiled into rdma_transport (#2725);
#   - cross-host hip+rdma segments automatically choose RDMA (#2753);
#   - large MRs are chunked and registration failures propagate (#2644/#2869);
#   - chunked dma-buf offsets include each chunk's displacement (#3243).
#
# Keep no private source patches here. Build the upstream source, then verify the
# two artifact capabilities SGLang PD needs: HIP dma-buf registration and
# cross-host RDMA routing.
set -euo pipefail

MC_ROOT="${MC_ROOT:-/tmp/mooncake-upstream}"
MC_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MC_REF="${MOONCAKE_GIT_REF:-faae8dd4a6309c3ecd47e0721a83b0250d686fa2}"
# gfx950 = MI355X; Dockerfile.sglang.gfx942 overrides this for MI300/MI325.
MC_GPU_ARCH="${MC_GPU_ARCH:-gfx950}"

rm -rf "$MC_ROOT"
git clone --filter=blob:none --no-checkout "$MC_REPO" "$MC_ROOT"
cd "$MC_ROOT"
git fetch --depth=1 origin "$MC_REF"
git checkout --detach "$MC_REF"
git submodule update --init --recursive
echo "[mc-build] Mooncake $(git rev-parse HEAD)"

# Assert the pinned source has the features this image relies on. A ref bump that
# loses or moves either implementation must fail here, before an expensive build.
RT=mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp
grep -q "target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)" \
    mooncake-transfer-engine/src/CMakeLists.txt ||
    { echo "[mc-build] ERROR: upstream HIP dma-buf build wiring missing" >&2; exit 1; }
grep -q "isHipReachableTarget" mooncake-transfer-engine/src/multi_transport.cpp ||
    { echo "[mc-build] ERROR: upstream cross-host HIP locality routing missing" >&2; exit 1; }
# ionic caps max_mr_size at 2 GiB, so a KV buffer above it is registered as
# several MRs. Both halves must be there: the split itself, and adding each
# chunk's displacement to its dma-buf offset. Without the second, every chunk
# past the first registers the wrong bytes.
grep -q "chunk_limit" "$RT" ||
    { echo "[mc-build] ERROR: upstream large-MR chunking missing (#2644)" >&2; exit 1; }
grep -q "chunk_dmabuf_exp.offset += chunk_offset" "$RT" ||
    { echo "[mc-build] ERROR: upstream per-chunk dma-buf offset missing (#3243)" >&2; exit 1; }

# docker build has no GPU, so pin the target architecture explicitly.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-$MC_GPU_ARCH}"
export GPU_ARCHS="${GPU_ARCHS:-$MC_GPU_ARCH}"
export AMDGPU_TARGETS="${AMDGPU_TARGETS:-$MC_GPU_ARCH}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-$MC_GPU_ARCH}"
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm-7.2.0/lib/cmake:$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"

rm -rf build
mkdir build
cd build
if ! cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF \
    -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
    -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$MC_GPU_ARCH" -GNinja \
    >/tmp/mooncake-cmake.log 2>&1; then
    tail -80 /tmp/mooncake-cmake.log >&2
    exit 1
fi
grep -iE "dmabuf|hsa-runtime|error" /tmp/mooncake-cmake.log || true

target="engine.cpython-310-x86_64-linux-gnu.so"
if ! ninja "$target" >/tmp/mooncake-ninja.log 2>&1; then
    tail -80 /tmp/mooncake-ninja.log >&2
    exit 1
fi
tail -10 /tmp/mooncake-ninja.log

SO="$MC_ROOT/build/mooncake-integration/$target"
DEST="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
cp "$SO" "$DEST"

ASIO="$MC_ROOT/build/mooncake-common/libasio.so"
if [ -f "$ASIO" ]; then
    cp "$ASIO" "$(dirname "$DEST")/"
    cp "$ASIO" /usr/local/lib/
    ldconfig
fi

# Artifact checks: inspect the .so Python will actually load.
dynsyms="$(nm -D "$DEST" 2>/dev/null || true)"
printf '%s\n' "$dynsyms" | grep -q "ibv_reg_dmabuf_mr" ||
    { echo "[mc-build] ERROR: dma-buf registration is absent from $DEST" >&2; exit 1; }
ldd "$DEST" | grep -qi hsa-runtime64 ||
    { echo "[mc-build] ERROR: $DEST does not link hsa-runtime64" >&2; exit 1; }
binary_strings="$(strings "$DEST")"
printf '%s\n' "$binary_strings" | grep -c "MC_DISABLE_HIP" | grep -qv '^0$' ||
    { echo "[mc-build] ERROR: cross-host HIP locality routing is absent" >&2; exit 1; }
python3 -c "from mooncake.engine import TransferEngine"

echo "[mc-build] DONE: upstream Mooncake $MC_REF"
echo "[mc-build] dma-buf=present cross-host-routing=present arch=$MC_GPU_ARCH"
cd /
rm -rf "$MC_ROOT"
