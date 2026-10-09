#!/usr/bin/env bash
# Criterion-1 evidence: patched bench without --profile vs the pristine copy, identical args.
set -euo pipefail
TOOL_ROOT=$(cd "$(dirname "$0")/.." && pwd)   # this tool's root; holds bench/
REPO=$(cd "$TOOL_ROOT/.." && pwd)            # repo root; bind-mounted into the container
BENCH_ROOT=${BENCH_ROOT:-$TOOL_ROOT}         # where bench/profile_decode.py lives
WS=${WS:-$PWD}                               # workspace for outputs; defaults to the cwd
PRISTINE=${PRISTINE:?set PRISTINE=/path/to/pre-feature/profile_decode.py}
ARGS=(--tp-size 8 --ep-size 8 --enable-dp-attention --batch-size 8 --max-running-requests 8
      --input-len 2048 --output-len 128 --accept-length 3.61 --warmup-steps 2 --max-steps 20
      --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85)
run() {  # run <tag> <entrypoint>
    local out="$WS/iterations/$1"
    [[ ! -e "$out" ]] || { echo "refusing existing $out" >&2; return 1; }
    mkdir -p "$out"
    docker exec -w / yihou-glm52-tp8ep8-0914 bash -lc \
        "python3 $2 --model-path /perf_apps/data/models/GLM-5.2-MXFP4 --result-dir $out ${ARGS[*]} > $out/runtime.log 2>&1"
    echo "$1 exit=$?"
}
run smoke_noprofile_patched_yihou "$BENCH_ROOT/bench/profile_decode.py"
run smoke_noprofile_pristine_yihou "$PRISTINE"
