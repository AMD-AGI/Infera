#!/usr/bin/env bash
# Run the fixed real-token correctness sequence (no simulated acceptance).
# Usage: bash run_correctness.sh
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$DIR/config.sh"
acquire_bench_lock

(( $# == 0 )) || {
    echo "usage: $0" >&2
    echo "run eval/smoke.sh, eval/long_context.sh, or eval/gsm8k.sh directly for one case" >&2
    exit 64
}

export RUN_ID
# config.sh defaults performance runs to simulated acceptance. An exported
# empty value survives child config.sh sourcing and disables it for this run.
export SIMULATE_ACC_LEN=
ROOT="$RUN_ROOT/correctness"
mkdir -p "$ROOT"

run_logged() {
    local name="$1"
    shift
    local rc
    set +e
    "$@" 2>&1 | tee "$ROOT/$name.log"
    rc=${PIPESTATUS[0]}
    set -e
    if (( rc != 0 )); then
        echo "[correctness] $name failed with rc=$rc; deployment left running" >&2
    fi
    return "$rc"
}

# Compute nodes currently mount /shared_nfs read-only. The mode probe is
# node-local; the two-rank fabric probe needs a writable shared dump and is
# therefore covered by the prior same-node result plus the live RDMA smoke below.
run_logged preflight env \
    PREFLIGHT_OUT_DIR="$ROOT/preflight" RUN_ID="$RUN_ID" \
    bash "$DIR/preflight.sh" mode
run_logged launch env RUN_ID="$RUN_ID" bash "$DIR/launch.sh"
run_logged smoke env \
    SMOKE_OUT_DIR="$ROOT/smoke" RUN_ID="$RUN_ID" \
    bash "$DIR/eval/smoke.sh"
run_logged long-context env \
    LONG_CONTEXT_OUT_DIR="$ROOT/long-context" RUN_ID="$RUN_ID" \
    bash "$DIR/eval/long_context.sh"
run_logged gsm8k env \
    GSM8K_OUT_DIR="$ROOT/gsm8k" RUN_ID="$RUN_ID" \
    bash "$DIR/eval/gsm8k.sh"
run_logged stop env RUN_ID="$RUN_ID" bash "$DIR/stop.sh"

echo "[correctness] complete: $ROOT"
