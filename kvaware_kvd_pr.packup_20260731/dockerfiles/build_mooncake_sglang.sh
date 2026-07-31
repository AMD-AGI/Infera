#!/usr/bin/env bash
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Rebuild the sglang base image's bundled Mooncake ONE TIME, IN PLACE, so that a
# single engine.so supports EVERY PD KV-transfer configuration and the choice is
# made entirely at RUNTIME (no per-feature image variant):
#
#   * single-node  vs  cross-node PD        — decided by how you launch the legs
#   * dma-buf GPUDirect (ibv_reg_dmabuf_mr)  — the DEFAULT registration path
#   * bare ibv_reg_mr + peermem              — MOONCAKE_DISABLE_HIP_DMABUF=1
#   * HIP intra-node P2P transport           — opt-in via MC_ENABLE_HIP_TRANSPORT=1
#
# WHY this is a single in-place build (not two images):
#   The base bundles Mooncake at upstream #2682 (commit 01d1eb2a), whose
#   rdma_context.cpp ALREADY carries a full RUNTIME dma-buf switch:
#     - hipMemoryTypeDevice + kernel support -> ibv_reg_dmabuf_mr (GPUDirect),
#     - MOONCAKE_DISABLE_HIP_DMABUF=1 (or host/managed mem, or a kernel lacking
#       CONFIG_PCI_P2PDMA / CONFIG_DMABUF_MOVE_NOTIFY) -> bare ibv_reg_mr.
#   So "compile dma-buf in, decide at runtime" needs NOTHING but compiling that
#   branch. #2682 has TWO defects that stop it, both fixed here:
#     (a) CMake propagation bug: USE_HIP_DMABUF is defined only on the
#         transfer_engine target, but the ibv_reg_dmabuf_mr call lives in
#         rdma_context.cpp -> the rdma_transport OBJECT lib. Without the define
#         there the #elif USE_HIP_DMABUF branch compiles out and registration
#         SILENTLY falls back to bare ibv_reg_mr (the dma-buf path is dead code).
#         Fix: propagate USE_HIP_DMABUF + hsa-runtime64 to rdma_transport.
#     (b) it installs a HIP IPC transport UNCONDITIONALLY and selectTransport
#         prefers it over RDMA, so CROSS-NODE PD breaks (hipIpcOpenMemHandle
#         cannot open a peer's handle). Fix: gate it behind MC_ENABLE_HIP_TRANSPORT
#         (default OFF) — the reusable in-tree B-group C++ patch, shared verbatim
#         with the vLLM build (deploy/docker/patches/mooncake_cpp), so there is no
#         duplicated hand-written source edit.
#
# Reuse + dedup: the HIP-transport gate and the auto-chunk-MR correctness fix come
# from the SAME deploy/docker/patches/mooncake_cpp diffs the vLLM image applies via
# build_mooncake_rocm.sh; this script only adds the dma-buf CMake propagation on
# top and flips the build to -DUSE_HIP_DMABUF=ON. Both diffs are `git apply --check`
# CLEAN on #2682.
#
# Idempotent + self-verifying: skips already-applied steps and FAILS the build if
# the rebuilt object does not actually reference ibv_reg_dmabuf_mr / link hsa.
set -euo pipefail

MC_ROOT="${MC_ROOT:-/sgl-workspace/Mooncake}"
PATCH_DIR="${MC_CPP_PATCH_DIR:-/tmp/mooncake_cpp}"
RCM="$MC_ROOT/mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt"

[ -f "$MC_ROOT/mooncake-transfer-engine/src/CMakeLists.txt" ] \
    || { echo "[mc-build] ERR: $MC_ROOT is not a Mooncake tree — base image layout changed; re-anchor" >&2; exit 1; }
[ -f "$PATCH_DIR/apply_mooncake_cpp_patches.sh" ] \
    || { echo "[mc-build] ERR: $PATCH_DIR/apply_mooncake_cpp_patches.sh missing — COPY the mooncake_cpp patch dir" >&2; exit 1; }

# ---- [1/4] reusable in-tree B-group C++ patches (HIP gate + auto-chunk MR) ----
# Shared verbatim with the vLLM build — no duplicated source edit. The gate makes
# HIP transport default-OFF so cross-node PD stays on RDMA; auto-chunk MR splits
# buffers larger than the device max_mr_size into <=max_mr_size MRs.
echo "=== [1/4] apply reusable mooncake_cpp B-group patches ==="
MC_ROOT="$MC_ROOT" bash "$PATCH_DIR/apply_mooncake_cpp_patches.sh"

# ---- [2/4] propagate USE_HIP_DMABUF to the rdma_transport OBJECT lib -----------
# rdma_context.cpp (the ibv_reg_dmabuf_mr call site) compiles here; upstream only
# defines USE_HIP_DMABUF on transfer_engine, so add it (and the hsa-runtime64 link
# the export helper needs) to this target too. Idempotent.
echo "=== [2/4] propagate USE_HIP_DMABUF -> rdma_transport ==="
if grep -q 'USE_HIP_DMABUF' "$RCM"; then
    echo "  already propagated — skip"
else
    cat >> "$RCM" <<'CM'

# infera(unified sglang): rdma_context.cpp calls ibv_reg_dmabuf_mr but lives in
# this OBJECT lib; upstream #2682 defines USE_HIP_DMABUF only on the
# transfer_engine target, so the #elif USE_HIP_DMABUF branch compiles out here and
# device-KV registration silently falls back to bare ibv_reg_mr. Mirror the base's
# transfer_engine block: define USE_HIP_DMABUF + link hsa-runtime64 so the dma-buf
# path is actually compiled. Runtime still decides bare vs dma-buf (see
# rdma_context.cpp: MOONCAKE_DISABLE_HIP_DMABUF / kernel-support probe).
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

# ---- [3/4] cmake configure + ninja the engine module (USE_HIP_DMABUF=ON) -------
# `docker build` has NO GPU, so ROCm's amdgpu-arch probe fails and cmake would
# drop the HIP engine target from the graph (leaving the base's unpatched .so).
# Pin the target arch explicitly so the build is GPU-independent. gfx950 = MI355X.
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"
export GPU_ARCHS="${GPU_ARCHS:-gfx950}" AMDGPU_TARGETS="${AMDGPU_TARGETS:-gfx950}"
export HIP_ARCHITECTURES="${HIP_ARCHITECTURES:-gfx950}"
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm-7.2.0/lib/cmake:$(python3 -c 'import pybind11;print(pybind11.get_cmake_dir())' 2>/dev/null):${CMAKE_PREFIX_PATH:-}"
echo "=== [3/4] cmake configure + ninja (USE_HIP=ON USE_HIP_DMABUF=ON) ==="
cd "$MC_ROOT"
rm -rf build && mkdir build && cd build
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
    -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_STORE_RUST=OFF \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja 2>&1 \
    | grep -iE "dmabuf|hsa-runtime|hip transport|error" | head -20 || true
# Build the pybind engine module explicitly — plain `ninja` builds only the
# default target set, which does NOT include it, so the prebuilt (unpatched) one
# would survive. Target it by name.
ninja engine.cpython-310-x86_64-linux-gnu.so 2>&1 | tail -8

# ---- [4/4] install over the pip engine.so + verify all three properties -------
echo "=== [4/4] install + verify (HIP gate + dma-buf compiled-in + hsa link) ==="
SO="$(ls "$MC_ROOT"/build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so | head -1)"
DEST="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
cp "$SO" "$DEST"
ASIO="$(ls "$MC_ROOT"/build/mooncake-common/libasio.so 2>/dev/null | head -1 || true)"
[ -n "$ASIO" ] && { cp "$ASIO" "$(dirname "$DEST")/" 2>/dev/null || true; cp "$ASIO" /usr/local/lib/ 2>/dev/null || true; ldconfig 2>/dev/null || true; }

# (a) HIP-transport gate present (grep into a var, not `strings | grep -q`: under
#     pipefail, grep -q closing the pipe early SIGPIPEs strings -> false negative).
if ! strings "$DEST" | grep -c MC_ENABLE_HIP_TRANSPORT | grep -qv '^0$'; then
    echo "[mc-build] ERROR: rebuilt engine.so lacks the HIP-transport gate — patch/build did not take" >&2
    exit 1
fi
echo "  HIP_GATE=yes"

# (b) dma-buf actually compiled into rdma_context.o (EXTERNAL symbols, so inspect
#     the object's undefined refs — not `strings`).
OBJ="$MC_ROOT/build/mooncake-transfer-engine/src/transport/rdma_transport/CMakeFiles/rdma_transport.dir/rdma_context.cpp.o"
if [ -f "$OBJ" ] && nm "$OBJ" 2>/dev/null | grep -qiE "ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf"; then
    echo "  DMABUF_COMPILED_IN=yes ($(nm "$OBJ" | grep -ioE 'ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf' | sort -u | tr '\n' ' '))"
else
    echo "[mc-build] ERROR: ibv_reg_dmabuf_mr NOT compiled into rdma_context.o — CMake propagation did not take" >&2
    exit 1
fi

# (c) engine.so links hsa-runtime64 (the export helper's runtime dep).
if ldd "$DEST" 2>/dev/null | grep -qi hsa-runtime64; then
    echo "  LINKS_HSA=yes"
else
    echo "[mc-build] ERROR: engine.so not linked to hsa-runtime64 — dma-buf export would fail at runtime" >&2
    exit 1
fi

python3 -c "from mooncake.engine import TransferEngine" || { echo "[mc-build] import failed" >&2; exit 1; }
echo "[mc-build] DONE — one engine.so: HIP gated (MC_ENABLE_HIP_TRANSPORT), dma-buf"
echo "           compiled-in (default ibv_reg_dmabuf_mr; MOONCAKE_DISABLE_HIP_DMABUF=1"
echo "           forces bare ibv_reg_mr). Runtime decides every path."

# trim the build tree to keep the image lean
rm -rf "$MC_ROOT/build"
