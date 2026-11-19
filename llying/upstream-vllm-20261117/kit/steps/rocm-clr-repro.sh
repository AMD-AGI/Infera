#!/usr/bin/env bash
# ROCm clr issue (U2 root cause): hipMemcpyBatchAsync on memory imported with hipImportExternalMemory faults on device
# 0 when the import lives on another device. One process per case (a fault kills the process); records the ROCm / HIP
# versions of the image for the issue text. Exit 0 when the result matches the known bug (devices 1 and 3 fault,
# device 0 and plain hipMalloc are fine) -- that is the reproduction; the issue text quotes the output.
# key: sha256sum /home/liyingli/bench_agentx/upstream-20261119/checks/rocm-clr/batch_ext_repro.c
. /home/liyingli/bench_agentx/upstream-20261119/jobs/lib.sh
need_ready v034 || exit 1
docker rm -f up-clr > /dev/null 2>&1
# shellcheck disable=SC2046
docker run --rm --name up-clr $(gpu_args) -v "$U/checks/rocm-clr:/src:ro" -v "$OUT:/out" --entrypoint bash "$IMG_V034" -c '
R=${ROCM_PATH:-/opt/rocm}
{ echo "image: '"$IMG_V034"'"; cat $R/.info/version* 2>/dev/null; hipconfig --version 2>/dev/null; echo;
  grep -m1 -h "HIP_VERSION_GITHASH\|define HIP_VERSION " $R/include/hip/hip_version.h 2>/dev/null; } > /out/versions.txt
gcc -O2 -D__HIP_PLATFORM_AMD__ -I$R/include /src/batch_ext_repro.c -o /tmp/repro -L$R/lib -lamdhip64 -lhsa-runtime64 \
    -Wl,-rpath,$R/lib || exit 2
for c in "0 external" "1 external" "3 external" "1 plain"; do
    echo "== $c"; timeout 120 /tmp/repro $c 2>&1 | tail -4; echo "rc=$?"
done' | tee "$OUT/repro.txt"
cat "$OUT/versions.txt"
# Expected bug signature: imports on devices 1 and 3 fault in __amd_rocclr_copyBufferBatch on GPU index 0.
[[ $(grep -c "copyBufferBatch" "$OUT/repro.txt") -ge 2 ]] && grep -A2 "== 0 external" "$OUT/repro.txt" | grep -q "batch D2H ok"
