#!/usr/bin/env bash
# Run one internal-decode point inside the owned container, snapshotting exactly what ran.
# Port of the GLM-5.2 run_decode.sh: same image-digest + owner assertions, no-clobber, code snapshot + hashes + diff,
# launch_started.json before launch and launch_status.json after. Run it ON the GPU host, detached for anything long.
# Usage: [CONTAINER=...] [ARGS_FILES="presets/a.args ..."] [ENTRY=profile_decode.py] bash run_decode.sh [--dry-run] ITERATION [benchmark args...]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/lib_v3.sh"
DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# -ge 1 ]] || { printf 'Usage: bash %s [--dry-run] ITERATION [benchmark arguments...]\n' "$0" >&2; exit 2; }
ITERATION=$1; shift
[[ "$ITERATION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || v3_die "Invalid iteration name: $ITERATION"
PRESET_ARGS=()
for f in ${ARGS_FILES:-}; do
    [[ "$f" = /* ]] || f="$HERE/../$f"
    while IFS= read -r item; do PRESET_ARGS+=("$item"); done < <(v3_expand_args "$f")
done
OUT="$WORK_DIR/iterations/$ITERATION"
CTR_OUT="/out/iterations/$ITERATION"
# ENTRY=compare_server.py / check_compare_paths.py runs the reference tools with the same guarantees.
ENTRY=${ENTRY:-profile_decode.py}
[[ "$ENTRY" =~ ^(profile_decode|compare_server|check_compare_paths)\.py$ ]] || v3_die "unknown ENTRY $ENTRY"
INNER=(python3 "$TOOL_IN/bench/$ENTRY" --model-path "$MODEL" --result-dir "$CTR_OUT" "${PRESET_ARGS[@]}" "$@")
printf -v INNER_CMD '%q ' "${INNER[@]}"
# runtime.log carries long single-line progress bars: write it to a file, never pipe it through tail.
CMD=(docker exec -w "$CTR_OUT" "$CONTAINER" bash -lc "set -o pipefail; ${PREFIX:-}$INNER_CMD > $CTR_OUT/runtime.log 2>&1")
if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Benchmark command: %s%s\n' "${PREFIX:-}" "$INNER_CMD"
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi
[[ ! -e "$OUT" ]] || v3_die "Refusing existing iteration: $OUT"
v3_check_container
mkdir -p "$OUT"
[[ -n "${MKDIR_EXTRA:-}" ]] && mkdir -p "$OUT/$MKDIR_EXTRA"
v3_snapshot "$OUT" "$CTR_OUT" "$INNER_CMD"
[[ -n "${KNOBS_JSON:-}" ]] && printf '%s\n' "$KNOBS_JSON" > "$OUT/profile_run_knobs_yihou.json"
START=$(date +%s)
# Positive start marker: "no news" is then unambiguous between still running and never started.
printf '{"started_at":"%s","started_epoch":%s,"pid":%s,"iteration":"%s"}\n' \
    "$(date -Is)" "$START" "$$" "$ITERATION" > "$OUT/launch_started.json"
set +e
"${CMD[@]}" > "$OUT/console.log" 2>&1
STATUS=$?
set -e
printf '{"exit_code":%s,"launch_wall_seconds":%s%s}\n' "$STATUS" "$(( $(date +%s) - START ))" "${STATUS_EXTRA:-}" \
    > "$OUT/launch_status.json"
# GLM-5.2 handoff trap #10: arguments silently lost on the way in. Read them back from config_yihou.json.
if ! python3 "$HERE/arg_check.py" "$OUT" "${PRESET_ARGS[@]}" "$@"; then
    [[ "$STATUS" == 0 ]] && STATUS=3
fi
exit "$STATUS"
