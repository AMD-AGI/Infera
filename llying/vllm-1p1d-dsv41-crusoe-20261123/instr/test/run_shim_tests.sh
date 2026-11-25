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
for arg in "" fail; do
    UMBP_VMM_MIN_GB=1 LD_PRELOAD="$OUT/umbp_vmm_pool.so" "$T/t" $arg 2>&1 | tail -3
done
echo "shim tests passed; built $OUT/umbp_vmm_pool.so"
