#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
NAME=${1:?new iteration name}
shift
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || exit 2
[[ "$(squeue -j 131080 -h -o '%u %T %N')" == 'yihou RUNNING crsuse2-m2m-271' ]] || exit 1
OUT="$HERE/iterations/$NAME"
[[ ! -e "$OUT" ]] || exit 1
mkdir -p "$OUT"
cp -a "$ROOT/bench" "$OUT/bench_snapshot"
git -C "$ROOT" rev-parse HEAD > "$OUT/git_head.txt"
git -C "$ROOT" diff --binary > "$OUT/code.diff"
CMD=(python3 "$ROOT/bench/compare_server.py" --model-path /shared_nfs/models/GLM-5.2-MXFP4 --result-dir "$OUT" --tp-size 4 --ep-size 4 --input-len 10000 --output-len 500 --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope --mem-fraction-static 0.85 "$@")
printf -v COMMAND '%q ' "${CMD[@]}"
printf '%s\n' "$COMMAND" > "$OUT/command.txt"
printf -v LOG '%q' "$OUT/runtime.log"
REMOTE=(docker exec -w / yihou-glm52-compare-20260910 bash -lc "set -o pipefail; $COMMAND 2>&1 | tee $LOG")
printf -v REMOTE_CMD '%q ' "${REMOTE[@]}"
start=$(date +%s)
set +e
spur exec 131080 bash -lc "test \"\$(hostname)\" = crsuse2-m2m-271 && $REMOTE_CMD" 2>&1 | tee "$OUT/console.log"
status=${PIPESTATUS[0]}
set -e
printf '{"exit_code":%s,"wall_seconds":%s}\n' "$status" "$(( $(date +%s)-start ))" > "$OUT/launch_status.json"
exit "$status"
