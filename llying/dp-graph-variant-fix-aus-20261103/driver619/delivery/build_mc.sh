#!/bin/bash
# DRIVER619-20261113: build a Mooncake tree inside an SGLang image the way the production image's Mooncake was built
# (Infera deploy/docker/scripts/build_mooncake_sglang.sh: USE_HIP, HIP dma-buf, multi-protocol, no etcd / store).
# Usage in the container: build_mc.sh SRC_DIR OUT_DIR engine|tests
#   engine  OUT_DIR/pkg: the engine module, hip_gem_allocator.so, the tree's allocator.py and the shared libraries of
#           this build the engine links, i.e. what a derived image copies into the mooncake package
#   tests   also hip_gem_allocator_test, rdma_hip_kfd_pin_test and the TENT library (compile check of its changes),
#           hip_gem_pin_test and the existing tests of regress
#   regress only existing tests the PR must not change: HIP transport, HIP stream pool, TCP transport, transport unit
#           tests, KFD pin anchor (for an upstream tree without the PR)
# SRC_DIR must be readable by root (not on the root-squashed /home).
SRC=$1 OUT=$2 MODE=${3:-engine}
mkdir -p "$OUT/pkg"
rm -rf /tmp/mcb && cp -a "$SRC" /tmp/mcb && cd /tmp/mcb || exit 1
git config --global --add safe.directory '*'
export PYTORCH_ROCM_ARCH=gfx950 GPU_ARCHS=gfx950 AMDGPU_TARGETS=gfx950 HIP_ARCHITECTURES=gfx950
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:$(ls -d /opt/rocm-*/lib/cmake | head -1):$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
extra=(-DBUILD_UNIT_TESTS=OFF)
targets=(engine build_hip_gem_allocator)
REGRESS=(rdma_hip_kfd_pin_test hip_transport_test hip_stream_pool_test tcp_transport_test transport_uint_test)
if [ "$MODE" = tests ]; then
    extra=(-DBUILD_UNIT_TESTS=ON -DUSE_TENT=ON)
    targets+=(hip_gem_allocator_test tent hip_gem_pin_test "${REGRESS[@]}")
elif [ "$MODE" = regress ]; then
    extra=(-DBUILD_UNIT_TESTS=ON -DUSE_TENT=ON)
    targets=("${REGRESS[@]}")
fi
mkdir build && cd build
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF \
    "${extra[@]}" -DBUILD_EXAMPLES=OFF -DCMAKE_HIP_ARCHITECTURES=gfx950 -GNinja > "$OUT/cmake.log" 2>&1 ||
    { tail -40 "$OUT/cmake.log"; exit 1; }
ninja -k 0 "${targets[@]}" > "$OUT/ninja.log" 2>&1
rc=$?
grep -E "error|FAILED" "$OUT/ninja.log" | head -20
tail -3 "$OUT/ninja.log"
engine=$(find . -name "engine.cpython-*.so" | head -1)
if [ -n "$engine" ]; then
    cp "$engine" "$OUT/pkg/"
    ldd "$engine" | awk '$3 ~ /^\/tmp\/mcb\// {print $3}' | xargs -r cp -t "$OUT/pkg/"
    ldd "$engine" | grep "not found"
fi
find . -name hip_gem_allocator.so -exec cp {} "$OUT/pkg/" \;
cp /tmp/mcb/mooncake-integration/allocator.py "$OUT/pkg/"
for t in hip_gem_allocator_test hip_gem_pin_test libtent.a "${REGRESS[@]}"; do
    find . -name "$t" -type f -exec cp {} "$OUT/" \;
done
for t in hip_gem_allocator_test hip_gem_pin_test "${REGRESS[@]}"; do
    [ -f "$OUT/$t" ] && ldd "$OUT/$t" | awk '$3 ~ /^\/tmp\/mcb\// {print $3}' | xargs -r cp -t "$OUT/"
done
cp "$OUT/pkg/hip_gem_allocator.so" "$OUT/" 2>/dev/null
ls -la "$OUT" "$OUT/pkg"
exit $rc
