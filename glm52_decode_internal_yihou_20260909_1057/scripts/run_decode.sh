#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
JOB=126175
NODE=crsuse2-m2m-055
CONTAINER=yihou-glm52-internal-20260909
IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then
    DRY_RUN=1
    shift
fi
if [[ $# -lt 1 ]]; then
    printf 'Usage: bash %s [--dry-run] ITERATION [benchmark arguments...]\n' "$0" >&2
    exit 2
fi
ITERATION=$1
shift
if [[ ! "$ITERATION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid iteration name: %s\n' "$ITERATION" >&2
    exit 2
fi
OUT="$ROOT/iterations/$ITERATION"
INNER=(python3 "$ROOT/bench/profile_decode.py"
    --model-path /shared_nfs/models/GLM-5.2-MXFP4 --result-dir "$OUT" "$@")
printf -v INNER_CMD '%q ' "${INNER[@]}"
printf -v LIVE_LOG '%q' "$OUT/runtime.log"
CMD=(docker exec -w / "$CONTAINER" bash -lc "set -o pipefail; $INNER_CMD 2>&1 | tee $LIVE_LOG")
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Existing allocation: %s; required node: %s\n' "$JOB" "$NODE"
    printf 'Benchmark command: %s\n' "$INNER_CMD"
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi
STATE=$(squeue -j "$JOB" -h -o '%u %T %N')
if [[ "$STATE" != "yihou RUNNING $NODE" ]]; then
    printf 'Refusing unexpected allocation state: %s\n' "$STATE" >&2
    exit 1
fi
if [[ -e "$OUT" ]]; then
    printf 'Refusing existing iteration directory: %s\n' "$OUT" >&2
    exit 1
fi
mkdir -p "$OUT"
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"
printf '\n' >> "$OUT/command.txt"
printf -v REMOTE '%q ' "${CMD[@]}"
PRECHECK='set -euo pipefail; test "$(hostname)" = crsuse2-m2m-055; '
PRECHECK+='test "$(docker inspect -f '\''{{.Image}}'\'' yihou-glm52-internal-20260909)" = '
PRECHECK+="$IMAGE; "
PRECHECK+='test "$(docker inspect -f '\''{{index .Config.Labels "owner"}}'\'' yihou-glm52-internal-20260909)" = yihou; '
spur exec "$JOB" bash -lc "$PRECHECK $REMOTE" 2>&1 | tee "$OUT/console.log"
