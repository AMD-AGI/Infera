#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Rebuild the sglang base image's bundled Mooncake with the HIP dma-buf RDMA MR
# path ENABLED — i.e. device-memory KV registration goes through
# ibv_reg_dmabuf_mr (via hsa_amd_portable_export_dmabuf) instead of bare
# ibv_reg_mr. This is the ONLY registration path that gives GPUDirect on hosts
# WITHOUT a peer-memory kernel module (no nvidia_peermem / amdgpu peermem), where
# a bare ibv_reg_mr on a device pointer fails with EFAULT.
#
# IMPORTANT — where this may pin: on an RDMA NIC WITHOUT ODP (On-Demand Paging;
# e.g. AMD Pensando ionic), ibv_reg_dmabuf_mr forces the driver to PIN the whole
# registered region → the KV pool is duplicated in VRAM (and can exhaust a KFD
# resource → HIP-209). Use this image ONLY where the KV NIC HAS ODP (e.g.
# Mellanox mlx5, which does dynamic attach → no pin, no double). That is why
# infera's default build (deploy/docker/scripts/build_mooncake_rocm.sh) drops the
# dma-buf path; this image is the opt-in counterpart for the mlx5 case.
#
# What it does, in order:
#   1. Apply the in-tree B-group C++ patches (idempotent, self-checking):
#        - transfer_engine_impl.diff : gate installTransport("hip") behind
#          MC_ENABLE_HIP_TRANSPORT (default OFF) so CROSS-NODE PD stays on RDMA.
#        - rdma_auto_chunk_mr_2017.diff : split a buffer larger than the device
#          max_mr_size into <=max_mr_size MRs. No-op for mlx5 (unlimited
#          max_mr_size); a correctness fix for any finite-max_mr_size NIC. Both
#          verified to `git apply --check` clean on the base's Mooncake #2682.
#   2. Propagate USE_HIP_DMABUF to the rdma_transport target (the piece the
#      in-tree patches deliberately do NOT carry): upstream only defines
#      USE_HIP_DMABUF on the transfer_engine target, but rdma_context.cpp (which
#      contains the ibv_reg_dmabuf_mr call) lives in the rdma_transport OBJECT
#      lib — without the define there, the #elif USE_HIP_DMABUF branch compiles
#      out and registration silently falls back to bare ibv_reg_mr.
#   3. cmake -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON, ninja the engine module, install
#      it over the pip-installed one, and VERIFY ibv_reg_dmabuf_mr is actually
#      referenced by the built rdma_context object (fail the build otherwise).
set -euo pipefail

MC_ROOT="${MC_ROOT:-/sgl-workspace/Mooncake}"
PATCH_DIR="${MC_CPP_PATCH_DIR:-/tmp/mooncake_cpp}"
[ -f "$MC_ROOT/mooncake-transfer-engine/src/CMakeLists.txt" ] \
    || { echo "ERR: $MC_ROOT not a Mooncake tree — base image layout changed" >&2; exit 1; }

echo "=== [1/4] apply in-tree B-group C++ patches (HIP gate + auto-chunk MR) ==="
MC_ROOT="$MC_ROOT" bash "$PATCH_DIR/apply_mooncake_cpp_patches.sh"

echo "=== [2/4] propagate USE_HIP_DMABUF to the rdma_transport target ==="
RCM="$MC_ROOT/mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt"
if grep -q 'USE_HIP_DMABUF' "$RCM"; then
    echo "  already propagated — skip"
else
    cat >> "$RCM" <<'CM'

# infera(dmabuf image): rdma_context.cpp calls ibv_reg_dmabuf_mr but lives in this
# OBJECT lib; upstream only defines USE_HIP_DMABUF on the transfer_engine target,
# so the branch compiles out here and registration falls back to bare ibv_reg_mr.
# Add the define + hsa-runtime64 link so the dma-buf path is actually compiled.
if(USE_HIP)
  target_include_directories(rdma_transport PRIVATE ${HIP_INCLUDE_DIRS})
  option(USE_HIP_DMABUF "Enable HIP dmabuf RDMA MR registration" ON)
  if(USE_HIP_DMABUF)
    find_package(hsa-runtime64 CONFIG)
    if(hsa-runtime64_FOUND)
      target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
      target_link_libraries(rdma_transport PRIVATE hsa-runtime64::hsa-runtime64 hip::host)
      message(STATUS "rdma_transport: HIP dmabuf MR registration enabled")
    else()
      message(FATAL_ERROR "USE_HIP_DMABUF requested but hsa-runtime64 not found")
    endif()
  endif()
endif()
CM
    echo "  propagated to $RCM"
fi

echo "=== [3/4] cmake configure + ninja build engine module (USE_HIP_DMABUF=ON) ==="
# docker build has NO GPU, so pin the arch explicitly (amdgpu-arch probe would
# fail and cmake would drop the HIP engine target). gfx950 = MI355X.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
export GPU_ARCHS="${GPU_ARCHS:-gfx950}" AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx950}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx950}"
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm-7.2.0/lib/cmake:$(python3 -c 'import pybind11;print(pybind11.get_cmake_dir())' 2>/dev/null):${CMAKE_PREFIX_PATH:-}"
cd "$MC_ROOT"
rm -rf build && mkdir build && cd build
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
    -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_STORE_RUST=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja 2>&1 \
    | grep -iE "dmabuf|hsa-runtime|error" | head -20
ninja engine.cpython-310-x86_64-linux-gnu.so 2>&1 | tail -6

echo "=== [4/4] install over pip engine.so + verify dma-buf compiled in ==="
SO="$(ls "$MC_ROOT"/build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so | head -1)"
DEST="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
cp "$SO" "$DEST"
ASIO="$(ls "$MC_ROOT"/build/mooncake-common/libasio.so 2>/dev/null | head -1 || true)"
[ -n "$ASIO" ] && { cp "$ASIO" "$(dirname "$DEST")/"; cp "$ASIO" /usr/local/lib/ 2>/dev/null || true; ldconfig 2>/dev/null || true; }

# The registration calls are EXTERNAL symbols, not string literals — check the
# rdma_context object's undefined refs, not `strings`.
OBJ="$MC_ROOT/build/mooncake-transfer-engine/src/transport/rdma_transport/CMakeFiles/rdma_transport.dir/rdma_context.cpp.o"
if [ -f "$OBJ" ] && nm "$OBJ" 2>/dev/null | grep -qiE "ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf"; then
    echo "DMABUF_COMPILED_IN=yes ($(nm "$OBJ" | grep -ioE 'ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf' | sort -u | tr '\n' ' '))"
else
    echo "ERROR: ibv_reg_dmabuf_mr NOT compiled into rdma_context.o — build did not take" >&2
    exit 1
fi
ldd "$DEST" 2>/dev/null | grep -qi hsa-runtime64 && echo "LINKS_HSA=yes" || { echo "ERROR: engine.so not linked to hsa-runtime64" >&2; exit 1; }
python3 -c "from mooncake.engine import TransferEngine; print('MC_IMPORT_OK')"
echo "MOONCAKE_DMABUF_BUILD_DONE"
