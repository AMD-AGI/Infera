#!/usr/bin/env bash
# Run INSIDE the base sglang container. Rebuild the bundled Mooncake with the HIP
# dmabuf RDMA MR path ENABLED (ibv_reg_dmabuf_mr via hsa_amd_portable_export_dmabuf),
# plus the MC_ENABLE_HIP_TRANSPORT env gate so cross-node PD uses RDMA (not hip IPC).
#
# Why dmabuf ON here (vs infera dropping it): infera targets ionic (no ODP) where
# dmabuf pins the whole KV pool -> KFD exhaustion (HIP-209). We FORCE mlx5 (HAS ODP)
# where dmabuf = dynamic attach = no pin = no double = no exhaustion.
set -euo pipefail

MC_ROOT="${MC_ROOT:-/sgl-workspace/Mooncake}"
SRC="$MC_ROOT/mooncake-transfer-engine/src/transfer_engine_impl.cpp"
[ -f "$SRC" ] || { echo "ERR: $SRC missing"; exit 1; }

echo "=== [1/4] gate HIP transport behind MC_ENABLE_HIP_TRANSPORT ==="
if grep -q 'MC_ENABLE_HIP_TRANSPORT' "$SRC"; then
  echo "  already gated - skip"
else
  python3 - "$SRC" <<'PY'
import sys
f = sys.argv[1]; s = open(f).read()
old = ("#ifdef USE_HIP\n"
       "        // HIP transport handles intra-node GPU P2P via XGMI/IPC and can\n"
       "        // coexist with the cross-node transport (RDMA/TCP) selected above.\n"
       "        {\n"
       "            Transport* hip_transport =")
new = ("#ifdef USE_HIP\n"
       "        // HIP transport handles intra-node GPU P2P via XGMI/IPC and can\n"
       "        // coexist with the cross-node transport (RDMA/TCP) selected above.\n"
       "        // FIX: default OFF - cross-node peer can't open a hip IPC segment;\n"
       "        // opt in for intra-node P2P with MC_ENABLE_HIP_TRANSPORT=1.\n"
       "        if (getenv(\"MC_ENABLE_HIP_TRANSPORT\")) {\n"
       "            Transport* hip_transport =")
assert s.count(old) == 1, "anchor not found/unique - source drifted"
open(f,"w").write(s.replace(old,new,1)); print("  patched", f)
PY
fi

echo "=== [1b/4] propagate USE_HIP_DMABUF to rdma_transport target (B.1 fix) ==="
# rdma_context.cpp lives in the rdma_transport OBJECT lib, but upstream only adds
# USE_HIP_DMABUF to the transfer_engine target -> the #elif USE_HIP_DMABUF branch
# in rdma_context.cpp compiles OUT -> falls through to bare ibv_reg_mr. Add the
# define + hsa link to rdma_transport so ibv_reg_dmabuf_mr is actually compiled.
RCM="$MC_ROOT/mooncake-transfer-engine/src/transport/rdma_transport/CMakeLists.txt"
if grep -q 'USE_HIP_DMABUF' "$RCM"; then
  echo "  already propagated - skip"
else
  cat >> "$RCM" <<'CM'

# FIX: propagate HIP dmabuf so rdma_context.cpp compiles the ibv_reg_dmabuf_mr path.
if(USE_HIP)
  target_include_directories(rdma_transport PRIVATE ${HIP_INCLUDE_DIRS})
  option(USE_HIP_DMABUF "Enable HIP dmabuf RDMA MR registration" ON)
  if(USE_HIP_DMABUF)
    find_package(hsa-runtime64 CONFIG)
    if(hsa-runtime64_FOUND)
      target_compile_definitions(rdma_transport PRIVATE USE_HIP_DMABUF)
      target_link_libraries(rdma_transport PRIVATE hsa-runtime64::hsa-runtime64 hip::host)
      message(STATUS "rdma_transport: HIP dmabuf MR registration enabled")
    endif()
  endif()
endif()
CM
  echo "  propagated to $RCM"
fi

echo "=== [2/4] cmake configure with USE_HIP_DMABUF=ON ==="
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm-7.2.0/lib/cmake:$(python3 -c 'import pybind11,sys;print(pybind11.get_cmake_dir())' 2>/dev/null):${CMAKE_PREFIX_PATH:-}"
export PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 AMDGPU_TARGETS=gfx950 HIP_ARCHITECTURES=gfx950
cd "$MC_ROOT"
rm -rf build && mkdir build && cd build
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DWITH_STORE=OFF \
  -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF -DWITH_STORE_RUST=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja 2>&1 | grep -iE "dmabuf|hsa-runtime|error" | head
# must see BOTH "HIP dmabuf ... enabled" AND "rdma_transport: HIP dmabuf ... enabled"

echo "=== [3/4] ninja build engine module ==="
ninja engine.cpython-310-x86_64-linux-gnu.so 2>&1 | tail -8

echo "=== [4/4] install over pip engine.so + verify dmabuf present ==="
SO="$(ls "$MC_ROOT"/build/mooncake-integration/engine.cpython-*-x86_64-linux-gnu.so | head -1)"
DEST="$(python3 -c 'import mooncake.engine as e; print(e.__file__)')"
cp "$SO" "$DEST"
ASIO="$(ls "$MC_ROOT"/build/mooncake-common/libasio.so 2>/dev/null | head -1 || true)"
[ -n "$ASIO" ] && { cp "$ASIO" "$(dirname "$DEST")/" && cp "$ASIO" /usr/local/lib/ 2>/dev/null; ldconfig 2>/dev/null || true; }

echo "--- verify: ibv_reg_dmabuf_mr actually referenced by the built code ---"
# Correct check: the rdma_transport object must carry an undefined ref to
# ibv_reg_dmabuf_mr / hsa_amd_portable_export_dmabuf (these are external calls,
# NOT string literals - so grep strings is wrong). Check the object + the .so's
# NEEDED libs (must now link libhsa-runtime64).
OBJ="$(ls "$MC_ROOT"/build/mooncake-transfer-engine/src/transport/rdma_transport/CMakeFiles/rdma_transport.dir/rdma_context.cpp.o 2>/dev/null | head -1)"
DMABUF_OK=0
if [ -n "$OBJ" ] && nm "$OBJ" 2>/dev/null | grep -qiE "ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf"; then
  echo "DMABUF_IN_OBJ=yes ($(nm "$OBJ" | grep -ioE 'ibv_reg_dmabuf_mr|hsa_amd_portable_export_dmabuf' | sort -u | tr '\n' ' '))"
  DMABUF_OK=1
fi
if ldd "$DEST" 2>/dev/null | grep -qi hsa-runtime64; then
  echo "LINKS_HSA=yes"
else
  echo "LINKS_HSA=no"
fi
[ "$DMABUF_OK" = 1 ] && echo "DMABUF_PRESENT=yes" || { echo "DMABUF_PRESENT=NO -- build did not take"; exit 1; }
strings "$DEST" | grep -qi MC_ENABLE_HIP_TRANSPORT && echo "HIP_GATE=yes" || echo "HIP_GATE=NO"
python3 -c "from mooncake.engine import TransferEngine; print('MC_IMPORT_OK')"
echo "MC_DMABUF_BUILD_DONE"
