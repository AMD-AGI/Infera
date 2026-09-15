#!/usr/bin/env bash
# Run ONE profiled internal-decode point inside the owned container, snapshotting what ran.
#
# Modelled on ../../glm52_tp8ep8_dpa_c128_yihou_20260914-0423/scripts/run_decode.sh: same
# image-digest + owner assertions, same code snapshot, same launch_status.json, same refusal
# to clobber an existing iteration directory. Differences: the measurement knobs are pinned
# to the packup's C=256 EP8 point and --profile is added.
#
# A profiled run is NOT a performance measurement. TPOT / throughput from the emitted
# result_yihou.json must not be quoted as performance; use the non-profiled packup numbers.
#
# Usage:
#   bash run_profile_yihou.sh [--dry-run] ITERATION [extra benchmark args...]
# Environment overrides (all optional):
#   CONC=256 ISL=70000 OSL=10000 ACCEPT=3.61 MEMFRAC=0.85 WARMUP=10
#   MAX_STEPS=40 PROFILE_START=20 PROFILE_NUM=5 PROFILE_RANKS=0 EP_SIZE=8
#   GRAPH=on|off          off appends --disable-cuda-graph
#   GRAPH_PACKET_CAPTURE  when set, exported as DEBUG_CLR_GRAPH_PACKET_CAPTURE
#   CAPTURE_PROFILE=1     adds --enable-profile-cuda-graph --profile-graph-capture-trace
set -euo pipefail

WS=$(cd "$(dirname "$0")/.." && pwd)
REPO=$(cd "$WS/.." && pwd)
BENCH_ROOT=${BENCH_ROOT:-$REPO/glm52_decode_internal_yihou_20260909_1057}

CONTAINER=${CONTAINER:-yihou-glm52-tp8ep8-0914}
IMAGE=${IMAGE:-sha256:b5aa5bd3d828285bff8dd01bb92e6106f569f5b1eab5de24558790b1b10a5dc0}
MODEL=${MODEL:-/perf_apps/data/models/GLM-5.2-MXFP4}

# Measurement knobs: the packup's C=256 EP8 point, unchanged. Only --max-steps is reduced,
# which is what makes a profiling run affordable; it does not alter the measurement basis.
CONC=${CONC:-256}
ISL=${ISL:-70000}
OSL=${OSL:-10000}
ACCEPT=${ACCEPT:-3.61}
MEMFRAC=${MEMFRAC:-0.85}
WARMUP=${WARMUP:-10}
EP_SIZE=${EP_SIZE:-8}
MAX_STEPS=${MAX_STEPS:-40}
PROFILE_START=${PROFILE_START:-20}
PROFILE_NUM=${PROFILE_NUM:-5}
PROFILE_RANKS=${PROFILE_RANKS:-0}
GRAPH=${GRAPH:-on}

DRY_RUN=0
if [[ "${1:-}" == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ $# -lt 1 ]]; then
    printf 'Usage: [CONC=...] [GRAPH=on|off] bash %s [--dry-run] ITERATION [benchmark arguments...]\n' "$0" >&2
    exit 2
fi
ITERATION=$1; shift
[[ "$ITERATION" =~ ^[A-Za-z0-9][A-Za-z0-9_-]*$ ]] || { printf 'Invalid iteration name: %s\n' "$ITERATION" >&2; exit 2; }
[[ "$CONTAINER" =~ ^yihou-[A-Za-z0-9_-]+$ ]] || exit 2
(( CONC % 8 == 0 )) || { printf 'CONC must be divisible by dp_size=8: %s\n' "$CONC" >&2; exit 2; }
# MAX_STEPS=0 means "run the full OSL" (profile_decode.py's own semantics), so it always
# covers the window; only a positive bound has to reach past it.
(( MAX_STEPS == 0 || MAX_STEPS >= PROFILE_START + PROFILE_NUM )) || {
    printf 'MAX_STEPS=%s ends before the profile window closes at %s\n' "$MAX_STEPS" "$((PROFILE_START + PROFILE_NUM))" >&2; exit 2; }

OUT="${OUTPUT_ROOT:-$WS/iterations}/$ITERATION"

ARGS=(--tp-size 8 --ep-size "$EP_SIZE" --enable-dp-attention
      --batch-size "$CONC" --max-running-requests "$CONC"
      --input-len "$ISL" --output-len "$OSL" --accept-length "$ACCEPT"
      --warmup-steps "$WARMUP" --max-steps "$MAX_STEPS"
      --enable-aiter-allreduce-fusion --enable-fused-qk-norm-rope
      --mem-fraction-static "$MEMFRAC"
      --profile --profile-start-step "$PROFILE_START" --profile-num-steps "$PROFILE_NUM"
      --profile-ranks "$PROFILE_RANKS")
[[ "$GRAPH" == off ]] && ARGS+=(--disable-cuda-graph)
[[ "${CAPTURE_PROFILE:-0}" == 1 ]] && ARGS+=(--enable-profile-cuda-graph --profile-graph-capture-trace)

INNER=(python3 "$BENCH_ROOT/bench/profile_decode.py" --model-path "$MODEL" --result-dir "$OUT" "${ARGS[@]}" "$@")
printf -v INNER_CMD '%q ' "${INNER[@]}"
PREFIX=""
if [[ -n "${GRAPH_PACKET_CAPTURE:-}" ]]; then
    printf -v PREFIX 'export DEBUG_CLR_GRAPH_PACKET_CAPTURE=%q; ' "$GRAPH_PACKET_CAPTURE"
fi
if [[ "${CAPTURE_PROFILE:-0}" == 1 ]]; then
    # decode_cuda_graph_runner._post_process_after_profile() dumps
    # `cuda_graph_runner_memory_usage.pickle` into the CWD unconditionally. With the
    # usual `-w /` that is EPERM for a non-root user and kills capture outright, so
    # capture-profile runs get a writable CWD. All eight ranks write that one filename,
    # so the surviving pickle belongs to whichever rank wrote last; the per-rank
    # key_averages tables go to runtime.log and the capture traces are rank-namespaced.
    printf -v PREFIX '%scd %q; ' "$PREFIX" "$OUT/graph_capture_cwd"
fi
printf -v LIVE_LOG '%q' "$OUT/runtime.log"
# runtime.log carries multi-megabyte single-line tqdm bars: write it to a file, never a pipe to tail.
CMD=(docker exec -w / "$CONTAINER" bash -lc "set -o pipefail; ${PREFIX}$INNER_CMD > $LIVE_LOG 2>&1")

if [[ "$DRY_RUN" == 1 ]]; then
    printf 'Benchmark command: %s%s\n' "$PREFIX" "$INNER_CMD"
    printf '%q ' "${CMD[@]}"; printf '\n'; exit 0
fi

[[ ! -e "$OUT" ]] || { printf 'Refusing existing iteration: %s\n' "$OUT" >&2; exit 1; }

# Assert the container really is the pinned image and ours, before anything is measured.
test "$(docker inspect -f '{{.Image}}' "$CONTAINER")" = "$IMAGE"
test "$(docker inspect -f '{{index .Config.Labels "owner"}}' "$CONTAINER")" = "$USER"

mkdir -p "$OUT"
[[ "${CAPTURE_PROFILE:-0}" == 1 ]] && mkdir -p "$OUT/graph_capture_cwd"
printf '%q ' "${CMD[@]}" > "$OUT/command.txt"; printf '\n' >> "$OUT/command.txt"
git -C "$REPO" rev-parse HEAD > "$OUT/git_head.txt"
git -C "$REPO" diff --binary > "$OUT/code.diff"
sha256sum "$BENCH_ROOT"/bench/*.py > "$OUT/code_hashes.sha256"
cp -a "$BENCH_ROOT/bench" "$OUT/bench_snapshot"
docker inspect "$CONTAINER" > "$OUT/container-inspect.json"
printf '{"graph":"%s","conc":%s,"ep_size":%s,"max_steps":%s,"profile_start":%s,"profile_num":%s,"graph_packet_capture":"%s","capture_profile":"%s"}\n' \
    "$GRAPH" "$CONC" "$EP_SIZE" "$MAX_STEPS" "$PROFILE_START" "$PROFILE_NUM" \
    "${GRAPH_PACKET_CAPTURE:-unset}" "${CAPTURE_PROFILE:-0}" > "$OUT/profile_run_knobs_yihou.json"

START=$(date +%s)
# Positive start marker, not only a failure record: without this, "no news" from outside is
# indistinguishable between "still running" and "died before launch". launch_status.json only
# appears on exit, so a run that never starts leaves no trace in the iteration dir at all.
printf '{"started_at":"%s","started_epoch":%s,"pid":%s,"graph":"%s","conc":%s,"max_steps":%s,"profile_window":[%s,%s]}\n' \
    "$(date -Is)" "$START" "$$" "$GRAPH" "$CONC" "$MAX_STEPS" "$PROFILE_START" "$((PROFILE_START + PROFILE_NUM))" \
    > "$OUT/launch_started.json"
set +e
"${CMD[@]}" > "$OUT/console.log" 2>&1
STATUS=$?
set -e
printf '{"exit_code":%s,"launch_wall_seconds":%s,"is_performance_measurement":false}\n' \
    "$STATUS" "$(( $(date +%s) - START ))" > "$OUT/launch_status.json"
exit "$STATUS"
