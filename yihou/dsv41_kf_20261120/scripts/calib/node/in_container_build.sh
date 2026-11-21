#!/bin/bash
# Runs INSIDE a throwaway container of the 1106 image (no GPU). Builds the fork rocprofiler-sdk + json-tool
# into /yihou/local/calib_profiler/build (= $LCP/build on the node). Records the image facts the plan needs.
set -uo pipefail
SRC=$LCP/${SRC_SUB:?set by kf_build.sh}/projects/rocprofiler-sdk
B=$LCP/build
: "${CP:?}"
mkdir -p "$B"
echo "=== image facts"
echo ROCM_PATH=$ROCM_PATH; ls -la ${ROCM_PATH:-/opt/rocm}/lib/librocprofiler-sdk.so* ${ROCM_PATH:-/opt/rocm}/lib/librocprofiler-register.so* 2>&1
/opt/rocm/bin/rocprofv3 --help 2>&1 | grep -i -E 'attach|collection-period' | head -5
python3 -c "import sys,torch; print(sys.version); print(torch.__version__, torch.version.hip)"
python3 -c "import sitecustomize; print('image sitecustomize:', sitecustomize.__file__)" 2>&1 | tail -1
cmake --version | head -1; which hipcc g++ gcc; g++ --version | head -1
dpkg -l libdw-dev libelf-dev libdrm-dev 2>&1 | grep -E '^(ii|un)|no packages' || true
echo "=== deps"
need_cmake=$(python3 - <<'EOF'
import re,subprocess
try:
    v=re.search(r'(\d+)\.(\d+)',subprocess.check_output(['cmake','--version'],text=True)).groups()
    print(0 if (int(v[0]),int(v[1]))>=(3,24) else 1)
except Exception: print(1)
EOF
)
if [ "$need_cmake" = 1 ]; then
  python3 -m pip install --no-index --find-links "$CP/stage/wheels" 'cmake>=3.24' ninja || { echo "FAIL: cmake>=3.24 unavailable"; exit 2; }
  hash -r; cmake --version | head -1
fi
if ! dpkg -s libdw-dev >/dev/null 2>&1 || ! dpkg -s libelf-dev >/dev/null 2>&1 || ! dpkg -s libnuma-dev >/dev/null 2>&1; then
  timeout 600 apt-get update -qq && timeout 900 apt-get install -y -qq libdw-dev libelf-dev libnuma-dev \
    || { echo "FAIL: libdw-dev/libelf-dev missing and apt unavailable (stop condition S0-deps)"; exit 3; }
fi
echo "=== configure"
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_PREFIX_PATH="${ROCM_PATH:-/opt/rocm};${ROCM_PATH:-/opt/rocm}/lib/rocm_sysdeps" \
  -DCMAKE_INSTALL_PREFIX=$LCP/install \
  -DROCPROFILER_BUILD_TESTS=ON -DROCPROFILER_BUILD_SAMPLES=OFF -DROCPROFILER_BUILD_DOCS=OFF \
  -DROCPROFILER_BUILD_WERROR=OFF -DGPU_TARGETS=gfx950 -DCMAKE_HIP_ARCHITECTURES=gfx950 \
  > "$LCP/logs/cmake-configure.log" 2>&1 || { tail -60 "$LCP/logs/cmake-configure.log"; echo "FAIL: configure"; exit 4; }
echo "=== build json-tool (pulls in librocprofiler-sdk)"
cmake --build "$B" --target rocprofiler-sdk-json-tool --parallel "${JOBS:-32}" \
  > "$LCP/logs/build-jsontool.log" 2>&1 || { tail -80 "$LCP/logs/build-jsontool.log"; echo "FAIL: build"; exit 5; }
echo "=== optional: rocprofv3 tool library (best effort)"
cmake --build "$B" --target rocprofiler-sdk-tool --parallel "${JOBS:-32}" > "$LCP/logs/build-tool.log" 2>&1 \
  && echo "rocprofv3 tool built" || echo "rocprofv3 tool target not built (not needed for the json-tool path)"
echo "=== checks"
ls -la "$B/lib/librocprofiler-sdk.so"* "$B/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so"* || exit 6
nm -D "$B/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so" | grep -E 'json_tool_(start|stop|pause_until_graph|finalize_idle_worker)'
nm -D "$B/lib/librocprofiler-sdk.so" | grep rocprofiler_experimental_set_dispatch_strict_window
python3 "$SRC/tests/graph-attribution-experiments/validate_scoped_strict.py" "$B/lib/librocprofiler-sdk.so" \
  || echo "WARN: validate_scoped_strict did not PASS (strict window is not used in this plan)"
ldd "$B/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so" | grep -E 'rocprofiler|amdhip|hsa'
echo "${SRC_SHA256:?}" > "$B/SOURCE_SHA256"
sha256sum "$B/lib/librocprofiler-sdk.so" "$B/lib/rocprofiler-sdk/librocprofiler-sdk-json-tool.so" | tee "$B/BUILD_OK"
echo "source archive sha256 $SRC_SHA256 ($SRC_SUB)"
echo "BUILD DONE"
