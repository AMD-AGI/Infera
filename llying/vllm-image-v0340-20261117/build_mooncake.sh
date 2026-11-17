#!/usr/bin/env bash
# Rebuild the Mooncake transfer engine of the vLLM v0.34.0 ROCm image (PyPI mooncake-transfer-engine-rocm 0.3.14)
# from the v0.3.14 source plus two of our open upstream PRs, and replace the engine extension the Python package loads:
#   Mooncake#4540  MC_RDMA_WRITE_PREFER_DEST_NIC: a WRITE into a peer GPU goes through the HCA the peer's topology
#                  prefers for that GPU, same-named HCA on both ends. Needed on crusoe, where ionic_k only reaches
#                  ionic_k (one rail per NIC) and a same-host transfer is an RDMA loopback on one NIC.
#   Mooncake#4541  MC_IB_TIMEOUT: RC ACK timeout exponent (the receiving ionic drops frames under back-pressure).
# The PyPI 0.3.14 build has neither (checked 2026-11-17), so this is the vLLM counterpart of
# deploy/docker/scripts/build_mooncake_sglang.sh, whose private diffs do not apply to v0.3.14.
set -euo pipefail
MC_ROOT=${MC_ROOT:-/tmp/mooncake-src}
MC_REF=${MC_REF:-v0.3.14}
MC_GPU_ARCH=${MC_GPU_ARCH:-gfx950}
PATCH_DIR=${PATCH_DIR:-/tmp/mc-patches}
rm -rf "$MC_ROOT"
git clone --filter=blob:none --no-checkout https://github.com/kvcache-ai/Mooncake.git "$MC_ROOT"
cd "$MC_ROOT"
git fetch --depth=1 origin "refs/tags/$MC_REF:refs/tags/$MC_REF"
git checkout --detach "$MC_REF"
git submodule update --init --recursive --depth=1
echo "[mc-build] Mooncake $(git rev-parse HEAD) ($MC_REF)"
for p in "$PATCH_DIR"/*.diff; do git apply "$p"; echo "[mc-build] applied $(basename "$p")"; done
export PYTORCH_ROCM_ARCH=$MC_GPU_ARCH GPU_ARCHS=$MC_GPU_ARCH AMDGPU_TARGETS=$MC_GPU_ARCH HIP_ARCHITECTURES=$MC_GPU_ARCH
export CMAKE_PREFIX_PATH="/opt/rocm:/opt/rocm/lib/cmake:$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())'):${CMAKE_PREFIX_PATH:-}"
if [ ! -f /usr/local/lib/cmake/yalantinglibs/yalantinglibsConfig.cmake ]; then
    cmake -S extern/yalantinglibs -B /tmp/ylt-build -GNinja -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARK=OFF \
        -DBUILD_UNIT_TESTS=OFF > /tmp/ylt.log 2>&1 && cmake --build /tmp/ylt-build >> /tmp/ylt.log 2>&1 &&
        cmake --install /tmp/ylt-build >> /tmp/ylt.log 2>&1 || { tail -40 /tmp/ylt.log >&2; exit 1; }
    rm -rf /tmp/ylt-build
fi
mkdir -p build && cd build
cmake .. -DUSE_HIP=ON -DUSE_HIP_DMABUF=ON -DUSE_ETCD=OFF -DENABLE_MULTI_PROTOCOL=ON -DWITH_STORE=OFF \
    -DWITH_STORE_RUST=OFF -DWITH_EP=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DCMAKE_HIP_ARCHITECTURES="$MC_GPU_ARCH" -GNinja > /tmp/mooncake-cmake.log 2>&1 ||
    { tail -80 /tmp/mooncake-cmake.log >&2; exit 1; }
grep -iE "dmabuf|hsa-runtime|HIP support" /tmp/mooncake-cmake.log || true
target="engine$(python3 -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
ninja "$target" > /tmp/mooncake-ninja.log 2>&1 || { tail -80 /tmp/mooncake-ninja.log >&2; exit 1; }
SO=$(find "$MC_ROOT/build" -name "$target" | head -1)
DEST=$(python3 -c 'import mooncake, os; print(os.path.join(os.path.dirname(mooncake.__file__), "engine.so"))')
cp "$DEST" "$DEST.pypi-0.3.14"
cp "$SO" "$DEST"
ASIO=$(find "$MC_ROOT/build" -name libasio.so | head -1)
[ -n "$ASIO" ] && cp "$ASIO" "$(dirname "$DEST")/"
python3 - "$DEST" <<'EOF'
import sys
data = open(sys.argv[1], "rb").read()
need = [b"MC_RDMA_WRITE_PREFER_DEST_NIC", b"MC_IB_TIMEOUT", b"MC_DISABLE_HIP", b"ibv_reg_dmabuf_mr",
        b"hsa_amd_portable_export_dmabuf", b"MC_ENABLE_DEST_DEVICE_AFFINITY"]
missing = [n.decode() for n in need if n not in data]
if missing:
    sys.exit(f"[mc-build] ERROR: {sys.argv[1]} lacks {missing}")
print("[mc-build] engine.so has", ", ".join(n.decode() for n in need))
EOF
python3 -c "from mooncake.engine import TransferEngine; print('[mc-build] import ok')"
cd / && rm -rf "$MC_ROOT"
