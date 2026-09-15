#!/usr/bin/env bash
# TP4 / DP-attention high-concurrency points, one arm (EP on or EP off) per invocation.
#
# Usage:
#   JOB_ID=... NODE=... CONTAINER=... OUTPUT_ROOT=... EP_SIZE=1|4 \
#     bash run_highconc_yihou.sh C:MEMFRAC [C:MEMFRAC ...]
#
# MEMFRAC is required per point: at these concurrencies the KV pool must be
# resized deliberately, so the value is never defaulted or silently reused.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
: "${JOB_ID:?}" "${NODE:?}" "${CONTAINER:?}" "${OUTPUT_ROOT:?Set a unique iterations directory}"
: "${EP_SIZE:?Set EP_SIZE to 1 (EP off) or 4 (EP on)}"
[[ "$EP_SIZE" == "1" || "$EP_SIZE" == "4" ]] || { printf 'EP_SIZE must be 1 or 4\n' >&2; exit 2; }
[[ $# -gt 0 ]] || { printf 'Give at least one C:MEMFRAC point\n' >&2; exit 2; }

for point in "$@"; do
    [[ "$point" =~ ^([0-9]+):(0\.[0-9]+)$ ]] || { printf 'Malformed point: %s\n' "$point" >&2; exit 2; }
    concurrency=${BASH_REMATCH[1]}
    memfrac=${BASH_REMATCH[2]}
    (( concurrency % 4 == 0 )) || { printf 'C must divide by 4 (dp_size): %s\n' "$concurrency" >&2; exit 2; }
    tag="hc_ep${EP_SIZE}_c${concurrency}_mf${memfrac//./p}_mrr_yihou"
    # --max-running-requests is GLOBAL and is divided by attn_dp_size inside SGLang.
    # Without it the pinned SGLang prints "Max running requests is reset to 48 for
    # speculative decoding", giving 48/4 = 12 request slots per rank and capping the
    # local batch at 12 (global C at 48) regardless of how large the KV pool is.
    args=(--tp-size 4 --ep-size "$EP_SIZE" --enable-dp-attention --batch-size "$concurrency"
        --input-len 70000 --output-len 10000 --accept-length 3.61
        --warmup-steps 10 --enable-aiter-allreduce-fusion
        --enable-fused-qk-norm-rope --mem-fraction-static "$memfrac"
        --max-running-requests "$concurrency")
    printf '=== %s START %s ===\n' "$tag" "$(date -u +%FT%TZ)"
    # A point that cannot fit its KV pool is a documented outcome, not a reason to
    # abandon the remaining points: record the failure and continue the arm.
    if bash "$ROOT/scripts/run_decode.sh" "$tag" "${args[@]}"; then
        printf '=== %s OK %s ===\n' "$tag" "$(date -u +%FT%TZ)"
    else
        printf '=== %s FAILED (exit recorded in launch_status.json) %s ===\n' "$tag" "$(date -u +%FT%TZ)"
    fi
done
