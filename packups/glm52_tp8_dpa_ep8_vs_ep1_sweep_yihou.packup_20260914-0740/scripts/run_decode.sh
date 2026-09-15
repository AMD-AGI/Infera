#!/usr/bin/env bash
# Run one internal-decode point inside the owned container, snapshotting exactly what ran.
# Adapted from the packup's scripts/run_decode.sh: `spur exec "$JOB_ID"` and the squeue/node guards
# are dropped (bare-metal host); the image-digest and owner assertions on the container are kept.
set -euo pipefail

WS=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$WS/.." && pwd)
BENCH_ROOT=${BENCH_ROOT:-$REPO/glm52_decode_internal_yihou_20260909_1057}

CONTAINER=${CONTAINER:-yihou-glm52-tp8ep8-0914}
IMAGE=${IMAGE:-sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0}
MODEL=${MODEL:-/perf_apps/data/models/GLM-5.2-MXFP4}

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ $# -lt 1 ]]; then
    printf 'Usage: [CONTAINER=...] bash %s [--dry-run] ITERATION [benchmark arguments...]\n' "$0" >&2
    exit 2
fi
ITERATION=$1; shift
[[ "$ITERATION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf 'Invalid iteration name: %s\n' "$ITERATION" >&2; exit 2; }
[[ "$CONTAINER" =~ ^yihou-[A-Za-z0-9_-]+$ ]] || exit 2

OUT="${OUTPUT_ROOT:-$WS/iterations}/$ITERATION"
INNER=(python3 "$BENCH_ROOT/bench/profile_decode.py" --model-path "$MODEL" --result-dir "$OUT" "$@")
printf -v INNER_CMD '%q ' "${INNER[@]}"
printf -v LIVE_LOG '%q' "$OUT/runtime.log"
CMD=(docker exec -w / "$CONTAINER" bash -lc "set -o pipefail; $INNER_CMD 2>&1 | tee $LIVE_LOG")

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Benchmark command: %s\n' "$INNER_CMD"
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi

[[ ! -e "$OUT" ]] || { printf 'Refusing existing iteration: %s\n' "$OUT" >&2; exit 1; }

# Assert the container really is the pinned image and ours, before anything is measured.
test "$(docker inspect -f '{{.Image}}' "$CONTAINER")" = "$IMAGE"
test "$(docker inspect -f '{{index .Config.Labels "owner"}}' "$CONTAINER")" = "$USER"

mkdir -p "$OUT"
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/git_head.txt"
git -C "$REPO" diff --binary > "$OUT/code.diff"
sha256sum "$BENCH_ROOT"/bench/*.py > "$OUT/code_hashes.sha256"
cp -a "$BENCH_ROOT/bench" "$OUT/bench_snapshot"
docker inspect "$CONTAINER" > "$OUT/container-inspect.json"

START=$(date +%s)
set +e
"${CMD[@]}" > "$OUT/console.log" 2>&1
STATUS=$?
set -e
printf '{"exit_code":%s,"launch_wall_seconds":%s}\n' "$STATUS" "$(( $(date +%s) - START ))" > "$OUT/launch_status.json"
exit "$STATUS"
