#!/usr/bin/env bash
# Runs inside the engine image: build MORI's umbp_standalone_server from SRC (MORI v1.2.3.post1 plus our patch) with the
# options of the image's own MORI build (vLLM docker/Dockerfile.rocm_base -> MORI setup.py: ROCm, UMBP on, SPDK off),
# and write the binary plus the dynamic dependencies of both the new and the image's binary to OUT.
#   build_umbp_inner.sh SRC OUT
set -x
SRC=${1:?} OUT=${2:?}
cp -a "$SRC" /tmp/mori && cd /tmp/mori || exit 1
# The image keeps only gRPC / Protobuf runtimes; the build needs their headers and code generators.
command -v grpc_cpp_plugin > /dev/null || { apt-get update && apt-get install -y --no-install-recommends \
    libgrpc++-dev libprotobuf-dev protobuf-compiler protobuf-compiler-grpc ninja-build; } > "$OUT/apt.log" 2>&1
cmake --version; command -v hipcc ninja protoc grpc_cpp_plugin
ROCM=${ROCM_PATH:-/opt/rocm}
cmake -S . -B build -G Ninja -DUSE_ROCM=ON -DCMAKE_BUILD_TYPE=Release -DGPU_TARGETS=gfx950 \
    -DBUILD_UMBP=ON -DUSE_SPDK=OFF -DUSE_REDIS_BACKEND=OFF -DBUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
    -DWITH_MPI=OFF -DBUILD_TORCH_BOOTSTRAP=OFF -DBUILD_XLA_FFI_OPS=OFF -DBUILD_CCO_SDMA=OFF -DBUILD_PYBINDS=OFF \
    -DCMAKE_PREFIX_PATH="$ROCM;$ROCM/lib/rocm_sysdeps" || exit 2
cmake --build build --target umbp_standalone_server -j "$(nproc)" || exit 3
bin=$(find build -name umbp_standalone_server -type f -perm -u+x | head -1)
[ -n "$bin" ] || exit 4
cp "$bin" "$OUT/umbp_standalone_server"
ldd "$OUT/umbp_standalone_server" > "$OUT/ldd-new.txt" 2>&1
P=$(python3 -c 'import mori, os; print(os.path.dirname(mori.__file__))')
ldd "$P/umbp_standalone_server" > "$OUT/ldd-image.txt" 2>&1
strings "$OUT/umbp_standalone_server" | grep -c 'UMBP_DRAM_PINNED' > "$OUT/has-pinned-env.txt"
echo "built $(stat -c %s "$OUT/umbp_standalone_server") bytes; image binary at $P"
