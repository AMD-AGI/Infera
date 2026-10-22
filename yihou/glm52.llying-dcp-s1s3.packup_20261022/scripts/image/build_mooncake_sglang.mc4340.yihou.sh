#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Rebuild the SGLang image's Mooncake from one pinned upstream revision.
# (yihou copy for infera-sglang:v0522-d395c3b3-1020-mc4340: also applies our port of
#  kvcache-ai/Mooncake#4340 (e8062249) to faae8dd4 and asserts it in source and artifact;
#  the committed build_mooncake_sglang.sh is unchanged.)
#
# That revision already contains the fixes Infera used to carry locally:
#   - HIP dma-buf is compiled into rdma_transport (#2725);
#   - cross-host hip+rdma segments automatically choose RDMA (#2753);
#   - large MRs are chunked and registration failures propagate (#2644/#2869);
#   - chunked dma-buf offsets include each chunk's displacement (#3243).
#
# The only private source patches are the two RDMA diffs in patches/mooncake_cpp:
# destination-local rail selection and a configurable RC ACK timeout. Both keep
# upstream behavior unless MC_ENABLE_DEST_LOCAL_RAIL / MC_IB_TIMEOUT are set.
# Build, then verify the artifact capabilities SGLang PD needs: HIP dma-buf
# registration, cross-host RDMA routing and both patch switches.
set -euo pipefail

MC_ROOT="${MC_ROOT:-/tmp/mooncake-upstream}"
MC_REPO="${MOONCAKE_REPO:-https://github.com/kvcache-ai/Mooncake.git}"
MC_REF="${MOONCAKE_GIT_REF:-faae8dd4a6309c3ecd47e0721a83b0250d686fa2}"
# gfx950 = MI355X; Dockerfile.sglang.gfx942 overrides this for MI300/MI325.
MC_GPU_ARCH="${MC_GPU_ARCH:-gfx950}"
MC_CPP_PATCH_DIR="${MC_CPP_PATCH_DIR:-$(cd "$(dirname "$0")/../patches/mooncake_cpp" && pwd)}"

rm -rf "$MC_ROOT"
git clone --filter=blob:none --no-checkout "$MC_REPO" "$MC_ROOT"
cd "$MC_ROOT"
git fetch --depth=1 origin "$MC_REF"
git checkout --detach "$MC_REF"
git submodule update --init --recursive
echo "[mc-build] Mooncake $(git rev-parse HEAD)"
MC_ROOT="$MC_ROOT" bash "$MC_CPP_PATCH_DIR/apply_mooncake_cpp_patches.sh" \
    rdma_dest_local_rail.diff rdma_ack_timeout.diff \
    mooncake-pr4340-e8062249.faae8dd4-port.yihou.diff
# yihou: #4340 (HIP IPC relocation for sub-allocated buffers) must be in the source.
HT=mooncake-transfer-engine/src/transport/hip_transport/hip_transport.cpp
grep -q "findIpcAllocation" "$HT" && grep -q "kIpcHandleWithOffsetBytes" "$HT" ||
    { echo "[mc-build] ERROR: #4340 HIP IPC offset path missing from source" >&2; exit 1; }

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

# The v0.5.21 base installed yalantinglibs under /usr/local; the v0.5.22 base does
# not. Install it from the pinned submodule, as Mooncake's dependencies.sh does.
if [ ! -f /usr/local/lib/cmake/yalantinglibs/yalantinglibsConfig.cmake ]; then
    if ! { cmake -S extern/yalantinglibs -B /tmp/ylt-build -GNinja -DBUILD_EXAMPLES=OFF \
            -DBUILD_BENCHMARK=OFF -DBUILD_UNIT_TESTS=OFF &&
        cmake --build /tmp/ylt-build && cmake --install /tmp/ylt-build; } >/tmp/ylt.log 2>&1; then
        tail -40 /tmp/ylt.log >&2
        exit 1
    fi
    rm -rf /tmp/ylt-build
    echo "[mc-build] installed yalantinglibs from the Mooncake submodule"
fi

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
for feature in MC_ENABLE_DEST_LOCAL_RAIL MC_IB_TIMEOUT; do
    [[ "$binary_strings" == *"$feature"* ]] ||
        { echo "[mc-build] ERROR: $DEST lacks $feature" >&2; exit 1; }
done
# yihou: here-string, not `printf | grep -q` (pipefail + SIGPIPE false failure).
grep -q "extends past its allocation" <<<"$binary_strings" ||
    { echo "[mc-build] ERROR: #4340 code path absent from $DEST" >&2; exit 1; }
echo "[mc-build] #4340 HIP IPC offset path present in $DEST"
python3 -c "from mooncake.engine import TransferEngine"

echo "[mc-build] DONE: Mooncake $MC_REF + RDMA patches + #4340 port"
echo "[mc-build] dma-buf=present cross-host-routing=present arch=$MC_GPU_ARCH"
cd /
rm -rf "$MC_ROOT"
