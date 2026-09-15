#!/usr/bin/env bash
# Copy of ../../../scripts/run_decode.sh with ONE addition: DOCKER_ENV entries are
# injected into the `docker exec` as -e NAME=VALUE. Everything else -- the allocation
# check, the node denylist, the image-digest and owner re-assertion, the code/hash/git
# snapshot and launch_status.json -- is byte-for-byte the same behaviour.
#
# Why a copy and not an edit of the shared script: the shared run_decode.sh is part of
# the audited harness referenced by a published packup, and bash reads a script
# incrementally, so editing it while a sweep is mid-flight can corrupt a running point.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
IMAGE=sha256:b9a83742f631d7321c6b75aceb031a8b347c70f97c62ef8c62370c686853de7d
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ $# -lt 1 ]]; then
    printf 'Usage: JOB_ID=... NODE=... CONTAINER=... [DOCKER_ENV="A=1 B=2"] bash %s [--dry-run] ITERATION [args...]\n' "$0" >&2
    exit 2
fi
ITERATION=$1
shift
if [[ ! "$ITERATION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]]; then
    printf 'Invalid iteration name: %s\n' "$ITERATION" >&2; exit 2
fi
: "${JOB_ID:?Set an existing authorized JOB_ID}" "${NODE:?Set its NODE}" "${CONTAINER:?Set the owned CONTAINER}"
[[ "$JOB_ID" =~ ^[0-9]+$ && "$NODE" =~ ^crsuse2-m2m-[0-9]+$ && "$CONTAINER" =~ ^yihou-[A-Za-z0-9_-]+$ ]] || exit 2
case "$NODE" in crsuse2-m2m-234|crsuse2-m2m-036|crsuse2-m2m-249) printf 'Forbidden or peer-owned node: %s\n' "$NODE" >&2; exit 2;; esac
ENV_FLAGS=()
for entry in ${DOCKER_ENV:-}; do
    [[ "$entry" =~ ^[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_,:.=-]+$ ]] || { printf 'Rejecting DOCKER_ENV entry: %s\n' "$entry" >&2; exit 2; }
    ENV_FLAGS+=(-e "$entry")
done
OUT="${OUTPUT_ROOT:-$ROOT/iterations}/$ITERATION"
INNER=(python3 "$ROOT/bench/profile_decode.py" --model-path /shared_nfs/models/GLM-5.2-MXFP4 --result-dir "$OUT" "$@")
printf -v INNER_CMD '%q ' "${INNER[@]}"
printf -v LIVE_LOG '%q' "$OUT/runtime.log"
CMD=(docker exec -w / "${ENV_FLAGS[@]}" "$CONTAINER" bash -lc "set -o pipefail; $INNER_CMD 2>&1 | tee $LIVE_LOG")
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Existing allocation: %s; required node: %s\nBenchmark command: %s\n' "$JOB_ID" "$NODE" "$INNER_CMD"
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi
STATE=$(squeue -j "$JOB_ID" -h -o '%u %T %N')
[[ "$STATE" == "$USER RUNNING $NODE" ]] || { printf 'Refusing unexpected allocation state: %s\n' "$STATE" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { printf 'Refusing existing iteration: %s\n' "$OUT" >&2; exit 1; }
mkdir -p "$OUT"
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
printf '%s\n' "${DOCKER_ENV:-}" > "$OUT/docker_env_yihou.txt"
git -C "$ROOT" rev-parse HEAD > "$OUT/git_head.txt"
git -C "$ROOT" diff --binary > "$OUT/code.diff"
sha256sum "$ROOT"/bench/*.py > "$OUT/code_hashes.sha256"
cp -a "$ROOT/bench" "$OUT/bench_snapshot"
printf -v REMOTE '%q ' "${CMD[@]}"
PRECHECK="set -euo pipefail; test \"\$(hostname)\" = $NODE; "
PRECHECK+="test \"\$(docker inspect -f '{{.Image}}' $CONTAINER)\" = $IMAGE; "
PRECHECK+="test \"\$(docker inspect -f '{{index .Config.Labels \"owner\"}}' $CONTAINER)\" = $USER; "
START=$(date +%s)
set +e
spur exec "$JOB_ID" bash -lc "$PRECHECK $REMOTE" 2>&1 | tee "$OUT/console.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '{"exit_code":%s,"launch_wall_seconds":%s}\n' "$STATUS" "$(( $(date +%s) - START ))" > "$OUT/launch_status.json"
exit "$STATUS"
