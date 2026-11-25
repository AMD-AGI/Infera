#!/usr/bin/env bash
# Build instr/umbp_vmm_pool.so and test it against the fake HIP library (no GPU needed). Run inside the engine image
# on a compute node (probe_kfd.sbatch does); writes the built shim to OUT (default: a temporary directory).
#   bash instr/test/run_shim_tests.sh [OUT]
set -euo pipefail
INSTR=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-$(mktemp -d)}
T=$(mktemp -d)
gcc -shared -fPIC -O2 -Wall -Wextra -o "$OUT/umbp_vmm_pool.so" "$INSTR/umbp_vmm_pool.c" -ldl -lpthread
gcc -shared -fPIC -O2 -o "$T/libamdhip64.so" "$INSTR/test/fakehip.c"
gcc -O1 -o "$T/t" "$INSTR/test/test_umbp_vmm_pool.c" -L"$T" -lamdhip64 -Wl,-rpath,"$T"
# LD_LIBRARY_PATH, not only the RUNPATH: the image's LD_LIBRARY_PATH would otherwise load the real libamdhip64.
for arg in "" fail interleave; do
    numa=; [[ $arg = interleave ]] && numa=interleave
    out=$(UMBP_VMM_NUMA=$numa UMBP_VMM_MIN_GB=1 LD_LIBRARY_PATH="$T" LD_PRELOAD="$OUT/umbp_vmm_pool.so" "$T/t" $arg 2>&1) || {
        echo "$out"; echo "shim test '${arg:-takeover}' FAILED"; exit 1; }
    echo "$out" | tail -3
done
# The engine-side host allocation logger.
gcc -shared -fPIC -O2 -Wall -Wextra -o "$OUT/hostalloc_log.so" "$INSTR/hostalloc_log.c" -ldl -lpthread
gcc -O1 -o "$T/h" "$INSTR/test/test_hostalloc_log.c" -L"$T" -lamdhip64 -Wl,-rpath,"$T"
mkdir -p "$T/hl" && HOSTALLOC_LOG_DIR="$T/hl" LD_LIBRARY_PATH="$T" LD_PRELOAD="$OUT/hostalloc_log.so" "$T/h"
log=$(cat "$T"/hl/hostalloc-*.log)
echo "$log" | grep -q 'hipHostMalloc .* 2097152 ' && echo "$log" | grep -q 'hipHostRegister .* 131072 flags 0x2 err 99' &&
    echo "$log" | grep -q 'SUMMARY(exit) live 0 bytes' &&
    echo "$log" | grep -q 'hipHostRegister 2 calls' || { echo "$log"; echo "hostalloc_log test FAILED"; exit 1; }
echo "$log" | grep -E 'SUMMARY|^[0-9:.]+ hip' | tail -4
echo "shim tests passed; built $OUT/umbp_vmm_pool.so and $OUT/hostalloc_log.so"
