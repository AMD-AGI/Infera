#!/usr/bin/env bash
# Same points as run_highconc_yihou.sh, but executed with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True.
#
# Why: at C=160 the KV pool must be ~143 GiB, which leaves only ~5.5 GiB free, and
# CUDA-graph capture then fails with
#   "HIP out of memory. Tried to allocate 962.00 MiB ... 798.00 MiB is free ...
#    1.36 GiB is reserved by PyTorch but unallocated"
# i.e. the shortfall (~400 MiB) is smaller than the memory locked up by allocator
# fragmentation. expandable_segments is PyTorch's own remedy for exactly that, and is
# what the error message itself recommends.
#
# Points run here are tagged "_xs" so they are never silently mixed with the default
# allocator points. A C=128 control is meant to be run BOTH ways so the allocator's
# effect on the measurement is quantified rather than assumed.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${JOB_ID:?}" "${NODE:?}" "${CONTAINER:?}" "${OUTPUT_ROOT:?}" "${EP_SIZE:?}"
[[ "$EP_SIZE" == "1" || "$EP_SIZE" == "4" ]] || { printf 'EP_SIZE must be 1 or 4\n' >&2; exit 2; }
[[ $# -gt 0 ]] || { printf 'Give at least one C:MEMFRAC point\n' >&2; exit 2; }

export DOCKER_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

for point in "$@"; do
    [[ "$point" =~ ^([0-9]+):(0\.[0-9]+)$ ]] || { printf 'Malformed point: %s\n' "$point" >&2; exit 2; }
    concurrency=${BASH_REMATCH[1]}
    memfrac=${BASH_REMATCH[2]}
    (( concurrency % 4 == 0 )) || { printf 'C must divide by 4 (dp_size): %s\n' "$concurrency" >&2; exit 2; }
    tag="hc_ep${EP_SIZE}_c${concurrency}_mf${memfrac//./p}_mrr_xs_yihou"
    args=(--tp-size 4 --ep-size "$EP_SIZE" --enable-dp-attention --batch-size "$concurrency"
        --input-len 70000 --output-len 10000 --accept-length 3.61
        --warmup-steps 10 --enable-aiter-allreduce-fusion
        --enable-fused-qk-norm-rope --mem-fraction-static "$memfrac"
        --max-running-requests "$concurrency")
    printf '=== %s START %s ===\n' "$tag" "$(date -u +%FT%TZ)"
    if bash "$HERE/run_decode_env_yihou.sh" "$tag" "${args[@]}"; then
        printf '=== %s OK %s ===\n' "$tag" "$(date -u +%FT%TZ)"
    else
        printf '=== %s FAILED %s ===\n' "$tag" "$(date -u +%FT%TZ)"
    fi
done
