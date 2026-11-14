#!/bin/bash
# DRIVER619-20261113: build a Mooncake tree inside the production image the way the image's Mooncake was built
# (Infera deploy/docker/scripts/build_mooncake_sglang.sh: USE_HIP, HIP dma-buf, multi-protocol, no etcd / store).
# Usage in the container: build_mc.sh SRC_DIR OUT_DIR engine|tests
#   engine  engine.cpython-310-*.so + hip_gem_allocator.so (what the derived image swaps in)
#   tests   also hip_gem_allocator_test, rdma_hip_kfd_pin_test and the TENT library (compile check of its changes)
SRC=$1 OUT=$2 MODE=${3:-engine}
mkdir -p "$OUT"
rm -rf /tmp/mcb && cp -a "$SRC" /tmp/mcb && cd /tmp/mcb || exit 1
git config --global --add safe.directory '*'
export PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 AMDGPU_TARGETS=gfx950 HIP_ARCHITECTURES=gfx950
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:/opt/rocm-7.2.0/lib/cmake:$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
extra=(-DBUILD_UNIT_TESTS=OFF)
targets=(engine.cpython-310-x86_64-linux-gnu.so build_hip_gem_allocator)
if [ "$MODE" = tests ]; then
    extra=(-DBUILD_UNIT_TESTS=ON -DUSE_TENT=ON)
    targets+=(hip_gem_allocator_test rdma_hip_kfd_pin_test tent)
fi
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
    "${extra[@]}" -DBUILD_EXAMPLES=OFF -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja > "$OUT/cmake.log" 2>&1 ||
    { tail -40 "$OUT/cmake.log"; exit 1; }
ninja "${targets[@]}" > "$OUT/ninja.log" 2>&1
rc=$?
grep -E "error|FAILED" "$OUT/ninja.log" | head -20
tail -3 "$OUT/ninja.log"
find . \( -name "engine.cpython-310-x86_64-linux-gnu.so" -o -name hip_gem_allocator.so -o -name hip_gem_allocator_test \
    -o -name rdma_hip_kfd_pin_test -o -name libtent.a \) -exec cp {} "$OUT/" \;
ls -la "$OUT"
exit $rc
